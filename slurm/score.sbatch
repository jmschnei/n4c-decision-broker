#!/usr/bin/env bash
#SBATCH --job-name=n4c-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# Scores every generated query (CPU only). N4C_EXECUTE=1 additionally runs gold and generated
# queries on the public endpoints (needs internet on this node; see tools/eval/score.py).
set -euo pipefail
export N4C_ROOT="${N4C_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# Container mode: re-run this very script inside the Pyxis/Enroot image (see config.sh).
if [ -n "${N4C_CONTAINER_IMAGE:-}" ] && [ -z "${N4C_IN_CONTAINER:-}" ]; then
  # shellcheck source=slurm/config.sh
  source "$N4C_ROOT/slurm/config.sh"
  exec srun "${CONTAINER_ARGS[@]}" ${N4C_GRES:+--gres="$N4C_GRES"} --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    --export=ALL,N4C_IN_CONTAINER=1 bash "$N4C_ROOT/slurm/score.sbatch"
fi
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"
EXTRA=()
[ "${N4C_EXECUTE:-0}" = 1 ] && EXTRA+=(--execute)
[ -n "${N4C_QALD_ENDPOINT:-}" ] && EXTRA+=(--qald-endpoint "$N4C_QALD_ENDPOINT")
"$PY" -I "$N4C_ROOT/tools/eval/score.py" \
  --data "$DATA_NORM" --gen "$RESULTS/generation" --out "$RESULTS/scores" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== scores in $RESULTS/scores (summary.md, summary.json, item_scores.jsonl)"
