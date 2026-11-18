#!/usr/bin/env bash
#SBATCH --job-name=n4c-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# Same as `bash slurm/setup.sh`, for clusters that forbid heavy work on the login node.
# Needs a partition WITH internet access. Submit from the repository root:
#   sbatch [--partition=...] slurm/setup.sbatch
set -euo pipefail
export N4C_ROOT="${N4C_ROOT:-$SLURM_SUBMIT_DIR}"
bash "$N4C_ROOT/slurm/setup.sh"
