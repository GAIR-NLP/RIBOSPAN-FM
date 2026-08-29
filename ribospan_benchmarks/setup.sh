#!/usr/bin/env bash
# Copyright (c) 2026 RIBOSPAN Team Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Download public checkpoints into model_weights/ and create the conda
# environment. Pin list: envs/benchmark.yml. Do not `conda env create -f`
# that file; mamba-ssm / flash-attn need the flags below.

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

if conda env list | awk -v n="$ENV_NAME" '$1 == n { found = 1 } END { exit !found }'; then
  echo "skip existing env '$ENV_NAME'"
  exit 0
fi

echo "==> create $ENV_NAME (pins: $ROOT/envs/benchmark.yml)"
conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.12 pip
# shellcheck disable=SC1091
conda activate "$ENV_NAME"
fail() {
  conda deactivate || true
  conda env remove -y -n "$ENV_NAME"
  echo "failed to create '$ENV_NAME'" >&2
  exit 1
}
trap fail ERR

conda install -y --override-channels -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-nvcc=12.4.131 \
  || echo "warning: cuda-nvcc install failed; flash-attn may need a system CUDA toolkit"
python -m pip install --upgrade pip
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 transformers==4.38.0 tokenizers==0.15.2 numpy==1.26.4 \
  rna-fm==0.2.2 ml-collections==1.1.0 \
  pyyaml pandas pyarrow scikit-learn matplotlib seaborn tqdm einops scipy opentsne pytest \
  ninja packaging hydra-core omegaconf bitarray sacrebleu regex
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
trap - ERR
