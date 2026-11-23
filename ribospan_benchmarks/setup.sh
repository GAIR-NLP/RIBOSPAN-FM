#!/usr/bin/env bash
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="${BENCHMARK_ENV:-benchmark}"
WEIGHTS_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights-only) WEIGHTS_ONLY=1; shift ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    -h|--help) echo "Usage: ./setup.sh [--weights-only] [--env NAME]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

W="$ROOT/model_weights"

get() {
  local url=$1 dest=$2 want=${3:-}
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && ( -z "$want" || "$(stat -c%s "$dest")" == "$want" ) ]]; then
    echo "keep $dest"
    return
  fi
  echo "download $dest"
  local -a opts=(-L --fail --retry 5 --retry-delay 4 -C - --progress-bar)
  [[ -n "${HF_TOKEN:-}" ]] && opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
  curl "${opts[@]}" -o "${dest}.part" "$url"
  if [[ -n "$want" && "$(stat -c%s "${dest}.part")" != "$want" ]]; then
    echo "size mismatch: $dest" >&2
    exit 1
  fi
  mv "${dest}.part" "$dest"
}

hf() { get "https://huggingface.co/$1/resolve/main/$2" "$3" "${4:-}"; }

unpack_zip() {
  python3 - "$@" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

zip_path, dest, kind = sys.argv[1:]
dest = Path(dest)
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as archive:
    members = [item for item in archive.infolist() if not item.is_dir()]
    if kind == "ribogram":
        hits = [item for item in members if Path(item.filename).name == "RiboGRAM.pkl"]
        sized = [item for item in hits if item.file_size == 3389985992]
        chosen = sized[0] if len(sized) == 1 else (hits[0] if len(hits) == 1 else None)
        if chosen is None:
            raise SystemExit(f"expected one RiboGRAM.pkl in {zip_path}, found {[item.filename for item in hits]}")
        target = dest / "RiboGRAM.pkl"
        with archive.open(chosen) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
        print(f"extracted {chosen.filename} -> {target}")
    elif kind == "rnaverse":
        configs = [item for item in members if Path(item.filename).name == "config.json"]
        if not configs:
            raise SystemExit(f"{zip_path} has no config.json")
        configs.sort(key=lambda item: (Path(item.filename).parent.name.lower() != "rnaverse", len(item.filename)))
        parent = Path(configs[0].filename).parent
        prefix = "" if str(parent) == "." else parent.as_posix() + "/"
        for item in members:
            name = item.filename
            if prefix and not name.startswith(prefix):
                continue
            relative = name[len(prefix):] if prefix else name
            if not relative or "/" in relative:
                continue
            target = dest / relative
            with archive.open(item) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024)
            print(f"extracted {relative}")
    else:
        raise SystemExit(f"unknown archive kind: {kind}")
PY
}

echo "==> $W"
mkdir -p "$W/RIBOSPAN-1K-15" "$W/RIBOSPAN-1K-40" "$W/RIBOSPAN-10K-15" "$W/RIBOSPAN-10K-40"
hf cuhkaih/rnafm RNA-FM_pretrained.pth "$W/RNA-FM/RNA-FM_pretrained.pth" 1194424423
get "https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt?download=1" \
  "$W/RiNALMo/rinalmo_giga_pretrained.pt" 2603787622
for item in \
  "config.json:" \
  "generation_config.json:" \
  "pytorch_model.bin.index.json:" \
  "pytorch_model-00001-of-00002.bin:4978686958" \
  "pytorch_model-00002-of-00002.bin:1467887024"
do
  rel="${item%%:*}"
  hf genbio-ai/AIDO.RNA-1.6B-CDS "$rel" "$W/AIDO.RNA-CDS/$rel" "${item#*:}"
done
get "https://zenodo.org/records/20481998/files/HydraRNA_model.pt?download=1" \
  "$W/HydraRNA/HydraRNA_model.pt" 336791962
hf Ziyao1010/LIMBER LIMBER-macro/pytorch_model.bin "$W/LIMBER-macro/pytorch_model.bin" 2657362522
hf Ziyao1010/LIMBER LIMBER-macro/model_config.json "$W/LIMBER-macro/model_config.json" 2179
get "https://github.com/heqin-zhu/structRFM/releases/download/v0.0.8/structRFM_checkpoint.tar.gz" \
  "$W/structRFM/structRFM_checkpoint.tar.gz" 319391429
get "https://github.com/heqin-zhu/structRFM/releases/download/v0.0.8/structRFM_SSP_bpRNA1m.pt" \
  "$W/structRFM/structRFM_SSP_bpRNA1m.pt" 348179698
get "https://github.com/heqin-zhu/structRFM/releases/download/v0.0.8/structRFM_SSP_RNAStrAlign.pt" \
  "$W/structRFM/structRFM_SSP_RNAStrAlign.pt" 348179375
if [[ ! -f "$W/structRFM/structRFM_checkpoint/model.safetensors" ]]; then
  tar -xzf "$W/structRFM/structRFM_checkpoint.tar.gz" -C "$W/structRFM"
fi
SSP="$ROOT/benchmark_projects/StructRFM_Secondary_Structure_Benchmark/data/structrfm_release"
SSP_BPSEQ="$SSP/tasks_data/seqcls_ssp/data/ssp"
SSP_READY=1
for split in TR0 TS0 RNAStrAlign_bpseq archiveII_bpseq; do
  if [[ ! -d "$SSP_BPSEQ/$split" || -z "$(ls -A "$SSP_BPSEQ/$split" 2>/dev/null)" ]]; then
    SSP_READY=0
    break
  fi
done
if [[ "$SSP_READY" -eq 0 ]]; then
  TAR="$SSP/tasks_data.tar.gz"
  if [[ ! -f "$TAR" || "$(stat -c%s "$TAR")" != "3757158638" ]]; then
    # The structRFM release ships the archive in four parts.
    for item in 000:1073741824 001:1073741824 002:1073741824 003:535933166; do
      get "https://github.com/heqin-zhu/structRFM/releases/download/v0.0.8/tasks_data.tar.gz.part-${item%%:*}" \
        "$TAR.part-${item%%:*}" "${item#*:}"
    done
    cat "$TAR".part-00{0,1,2,3} > "$TAR"
    if [[ "$(stat -c%s "$TAR")" != "3757158638" ]]; then
      echo "size mismatch: $TAR" >&2
      exit 1
    fi
    rm -f "$TAR".part-00{0,1,2,3}
  fi
  tar -xzf "$TAR" -C "$SSP"
fi
if [[ ! -f "$W/RiboGRAM/RiboGRAM.pkl" || "$(stat -c%s "$W/RiboGRAM/RiboGRAM.pkl")" != "3389985992" ]]; then
  get "https://zenodo.org/records/23201637/files/2.%20Model%20Availability.zip?download=1" \
    "$W/RiboGRAM/Model_Availability.zip" 22084952473
  unpack_zip "$W/RiboGRAM/Model_Availability.zip" "$W/RiboGRAM" ribogram
  if [[ "$(stat -c%s "$W/RiboGRAM/RiboGRAM.pkl")" != "3389985992" ]]; then
    echo "size mismatch: $W/RiboGRAM/RiboGRAM.pkl" >&2
    exit 1
  fi
  rm -f "$W/RiboGRAM/Model_Availability.zip"
fi
if [[ ! -f "$W/RNAverse/config.json" || ! -f "$W/RNAverse/pytorch_model.bin" || "$(stat -c%s "$W/RNAverse/pytorch_model.bin")" != "605924294" ]]; then
  get "https://ndownloader.figshare.com/files/70070067" "$W/RNAverse/rnaverse.zip"
  unpack_zip "$W/RNAverse/rnaverse.zip" "$W/RNAverse" rnaverse
  if [[ "$(stat -c%s "$W/RNAverse/pytorch_model.bin")" != "605924294" ]]; then
    echo "size mismatch: $W/RNAverse/pytorch_model.bin" >&2
    exit 1
  fi
fi

[[ "$WEIGHTS_ONLY" -eq 1 ]] && exit 0

if ! command -v conda >/dev/null 2>&1 || [[ "$(type -t conda 2>/dev/null)" != "function" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found; weights are in $W" >&2
    exit 1
  fi
fi

CREATED=0
if conda env list | awk -v n="$ENV_NAME" '$1 == n { found = 1 } END { exit !found }'; then
  echo "use existing env '$ENV_NAME'"
else
  echo "==> create $ENV_NAME (pins: $ROOT/envs/benchmark.yml)"
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 pip
  CREATED=1
fi
# shellcheck disable=SC1091
conda activate "$ENV_NAME"
if [[ "$CREATED" -eq 1 ]]; then
  fail() {
    conda deactivate || true
    conda env remove -y -n "$ENV_NAME"
    echo "failed to create '$ENV_NAME'" >&2
    exit 1
  }
  trap fail ERR
fi

conda install -y --override-channels -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-nvcc=12.4.131 \
  || echo "warning: cuda-nvcc install failed; flash-attn may need a system CUDA toolkit"
python -m pip install --upgrade pip
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 transformers==4.38.0 tokenizers==0.15.2 numpy==1.26.4 \
  rna-fm==0.2.2 ml-collections==1.1.0 \
  pyyaml pandas 'pyarrow==17.0.0' scikit-learn matplotlib seaborn tqdm einops scipy opentsne pytest \
  ninja packaging hydra-core omegaconf bitarray sacrebleu regex \
  'huggingface_hub==0.36.2' 'datasets==2.21.0' 'biopython==1.88'
MAMBA_SKIP_CUDA_BUILD=TRUE python -m pip install mamba-ssm==2.2.2 --no-build-isolation
python - <<'PY'
from pathlib import Path
import mamba_ssm
Path(mamba_ssm.__file__).write_text('__version__ = "2.2.2"\n', encoding="utf-8")
PY
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation \
  || echo "warning: flash-attn install failed; HydraRNA load needs flash_attn 2.7.x"
"$ROOT/benchmark_projects/LIMBER_Secondary_Structure_Benchmark/setup.sh"
if [[ "$CREATED" -eq 1 ]]; then
  trap - ERR
fi
