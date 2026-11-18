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
export N4C_ROOT="${N4C_ROOT:-$SLURM_SUBMIT_DIR}"
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"
EXTRA=()
[ "${N4C_EXECUTE:-0}" = 1 ] && EXTRA+=(--execute)
[ -n "${N4C_QALD_ENDPOINT:-}" ] && EXTRA+=(--qald-endpoint "$N4C_QALD_ENDPOINT")
"$VENV/bin/python" -I "$N4C_ROOT/tools/eval/score.py" \
  --data "$DATA_NORM" --gen "$RESULTS/generation" --out "$RESULTS/scores" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== scores in $RESULTS/scores (summary.md, summary.json, item_scores.jsonl)"
