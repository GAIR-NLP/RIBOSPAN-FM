#!/usr/bin/env bash
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="${BENCHMARK_ENV:-benchmark}"
SRC="$ROOT/.build/src"
mkdir -p "$SRC"

CAUSAL_TAG="v1.5.0.post8"
CAUSAL_DIR="causal-conv1d-1.5.0.post8"
MAMBA_TAG="v2.2.2"
MAMBA_DIR="mamba-2.2.2"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  if ! command -v conda >/dev/null 2>&1 || [[ "$(type -t conda 2>/dev/null)" != "function" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
      eval "$("${CONDA_EXE}" shell.bash hook)"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
      echo "conda not found" >&2
      exit 1
    fi
  fi
  conda activate "$ENV_NAME"
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
NVIDIA_ROOT="$CONDA_PREFIX/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia"
if [[ -d "$NVIDIA_ROOT" ]]; then
  NVIDIA_INC="$(find "$NVIDIA_ROOT" -mindepth 2 -maxdepth 2 -type d -name include | paste -sd: -)"
  NVIDIA_LIB="$(find "$NVIDIA_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib | paste -sd: -)"
  export CPLUS_INCLUDE_PATH="${NVIDIA_INC}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export C_INCLUDE_PATH="${NVIDIA_INC}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export LIBRARY_PATH="${NVIDIA_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

fetch() {
  local url=$1 dest=$2
  if [[ -f "$dest" ]]; then
    echo "keep $dest"
    return
  fi
  echo "download $dest"
  curl -L --fail --retry 5 --retry-delay 4 -o "${dest}.part" "$url"
  mv "${dest}.part" "$dest"
}

fetch "https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/${CAUSAL_TAG}.tar.gz" \
  "$SRC/causal-conv1d-${CAUSAL_TAG}.tar.gz"
fetch "https://github.com/state-spaces/mamba/archive/refs/tags/${MAMBA_TAG}.tar.gz" \
  "$SRC/mamba-${MAMBA_TAG}.tar.gz"

rm -rf "$SRC/$CAUSAL_DIR" "$SRC/$MAMBA_DIR"
tar -xzf "$SRC/causal-conv1d-${CAUSAL_TAG}.tar.gz" -C "$SRC"
tar -xzf "$SRC/mamba-${MAMBA_TAG}.tar.gz" -C "$SRC"

python - "$SRC/$CAUSAL_DIR/setup.py" "$SRC/$MAMBA_DIR/setup.py" <<'PY'
import pathlib, sys

ARCHS = ("80", "89", "90")

def rewrite(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    skipping = False
    inserted = False
    for line in lines:
        if 'cc_flag.append("-gencode")' in line or "cc_flag.append('-gencode')" in line:
            skipping = True
            if not inserted:
                indent = line[: len(line) - len(line.lstrip())]
                block = []
                for arch in ARCHS:
                    block.append(f'{indent}cc_flag.append("-gencode")')
                    block.append(f'{indent}cc_flag.append("arch=compute_{arch},code=sm_{arch}")')
                out.extend(block)
                inserted = True
            continue
        if skipping:
            stripped = line.strip()
            if (
                stripped.startswith("cc_flag.append(")
                or stripped.startswith("if bare_metal_version")
                or stripped == ""
            ):
                continue
            skipping = False
        out.append(line)
    if not inserted:
        raise SystemExit(f"no -gencode block in {path}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"arches sm_{'/'.join(ARCHS)} in {path}")

for name in sys.argv[1:]:
    rewrite(pathlib.Path(name))
PY

python -m pip install "$SRC/$CAUSAL_DIR" --no-build-isolation --no-deps --force-reinstall --no-cache-dir
python -m pip install "$SRC/$MAMBA_DIR" --no-build-isolation --no-deps --force-reinstall --no-cache-dir

python - <<'PY'
import subprocess
import torch
from mamba_ssm import Mamba

mods = ["causal_conv1d", "causal_conv1d_cuda", "selective_scan_cuda", "mamba_ssm"]
loaded = []
for name in mods:
    mod = __import__(name)
    print(name, "ok", getattr(mod, "__file__", ""))
    loaded.append(mod)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to check the mamba kernels")
layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 32, 64, device="cuda", requires_grad=True)
y = layer(x)
y.sum().backward()
print("mamba forward/backward ok", tuple(y.shape))

needed = ("sm_80", "sm_89", "sm_90")
if subprocess.call(["bash", "-lc", "command -v cuobjdump >/dev/null"]) != 0:
    raise SystemExit("cuobjdump not found; cannot check cubins")
for mod in loaded:
    path = getattr(mod, "__file__", "")
    if not path.endswith(".so"):
        continue
    out = subprocess.check_output(["cuobjdump", "-lelf", path], text=True)
    missing = [arch for arch in needed if arch not in out]
    if missing:
        raise SystemExit(f"{path} missing {missing}")
    print(mod.__name__, "cubins", " ".join(needed))
PY
