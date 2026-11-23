#!/bin/bash
#SBATCH --job-name=rnacentral_stats
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rnacentral_stats_%j.out
#SBATCH --error=logs/rnacentral_stats_%j.err

# --------------------------------------------------------------------------- #
# RNAcentral analysis job for the Beacon cluster.
#
# Submit:  sbatch scripts/run_analyze_rnacentral.slurm
# Monitor: squeue -u $USER
# Cancel:  scancel JOBID
# --------------------------------------------------------------------------- #

set -euo pipefail

# Run from the project root (directory that contains scripts/ and data/).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "[slurm] job        : ${SLURM_JOB_ID:-NA}"
echo "[slurm] node       : $(hostname)"
echo "[slurm] started    : $(date -Is)"
echo "[slurm] workdir    : $(pwd)"

# Activate the conda environment.
# Adjust 'conda' bootstrap to match the cluster module / install if needed.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rnafm

# Full-dataset run (downloads the FASTA on first use if missing).
python scripts/analyze_rnacentral.py

echo "[slurm] finished   : $(date -Is)"
