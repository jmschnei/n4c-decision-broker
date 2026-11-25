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
  exec srun "${CONTAINER_ARGS[@]}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    --export=ALL,N4C_IN_CONTAINER=1 bash "$N4C_ROOT/slurm/score.sbatch"
fi
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"
EXTRA=()
if [ "${N4C_EXECUTE:-0}" = 1 ]; then
  EXTRA+=(--execute)
  # Wikidata (QALD) answers 403 to agents without contact info: score.py would abort.
  [ -n "${N4C_EVAL_USER_AGENT:-}" ] || [ -n "${N4C_QALD_ENDPOINT:-}" ] || echo \
    "WARNING: N4C_EXECUTE=1 without N4C_EVAL_USER_AGENT='n4c-eval/0.1 (mailto:you@institution)': Wikidata will refuse it"
fi
# Spider4SPARQL runs locally (Oxigraph, no internet) whenever its KGs were downloaded by setup.sh.
if [ -d "$DATA_RAW/spider4sparql_kg" ] && PYTHONPATH="${N4C_PYLIB:-}" "$PY" -c "import pyoxigraph" 2>/dev/null; then
  EXTRA+=(--spider-kg "$DATA_RAW/spider4sparql_kg")
else
  echo "NOTE: Spider4SPARQL scored offline only (needs $DATA_RAW/spider4sparql_kg and pyoxigraph)"
fi
[ -n "${N4C_QALD_ENDPOINT:-}" ] && EXTRA+=(--qald-endpoint "$N4C_QALD_ENDPOINT")
"$PY" -I "$N4C_ROOT/tools/eval/score.py" \
  --data "$DATA_NORM" --gen "$RESULTS/generation" --out "$RESULTS/scores" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== scores in $RESULTS/scores (summary.md, summary.json, item_scores.jsonl)"
