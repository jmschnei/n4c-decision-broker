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
export N4C_ROOT="${N4C_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# Container mode: re-run this very script inside the Pyxis/Enroot image (see config.sh).
if [ -n "${N4C_CONTAINER_IMAGE:-}" ] && [ -z "${N4C_IN_CONTAINER:-}" ]; then
  # shellcheck source=slurm/config.sh
  source "$N4C_ROOT/slurm/config.sh"
  exec srun "${CONTAINER_ARGS[@]}" ${N4C_GRES:+--gres="$N4C_GRES"} --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    --export=ALL,N4C_IN_CONTAINER=1 bash "$N4C_ROOT/slurm/setup.sbatch"
fi
bash "$N4C_ROOT/slurm/setup.sh"
