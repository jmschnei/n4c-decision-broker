#!/usr/bin/env bash
# Submit the evaluation: one GPU array job (models x shards) + a scoring job that runs afterwards.
#
#   bash slurm/submit.sh [--partition P] [--gres gpu:1] [--time HH:MM:SS] [--shards N]
#                        [--max-parallel K] [--run-name NAME] [--models "m1 m2"] [--limit N]
#                        [--execute] [--account A] [--dry-run]
set -euo pipefail
export N4C_ROOT="${N4C_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

PARTITION="" GRES="gpu:1" TIME="03:00:00" ACCOUNT="" THROTTLE="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --partition) PARTITION=$2; shift 2 ;;   --gres) GRES=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;             --account) ACCOUNT=$2; shift 2 ;;
    --shards) export N4C_SHARDS=$2; shift 2 ;;  --max-parallel) THROTTLE=$2; shift 2 ;;
    --run-name) export N4C_RUN_NAME=$2; shift 2 ;;  --models) export N4C_MODELS=$2; shift 2 ;;
    --limit) export N4C_LIMIT=$2; shift 2 ;;  --execute) export N4C_EXECUTE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"

if [ "$DRY" = 0 ]; then
  [ -x "$OLLAMA_DIR/bin/ollama" ] && [ -f "$DATA_NORM/eval_sample.jsonl" ] \
    || { echo "Setup not done in $WORK. Run first:  bash slurm/setup.sh"; exit 1; }
fi
NM=${#MODELS[@]}
LAST=$((NM * NUM_SHARDS - 1))
ARRAY="0-$LAST${THROTTLE:+%$THROTTLE}"
COMMON=(--parsable --export=ALL --chdir "$N4C_ROOT")
[ -n "$PARTITION" ] && COMMON+=(--partition "$PARTITION")
[ -n "$ACCOUNT" ] && COMMON+=(--account "$ACCOUNT")
LOGS="$RESULTS/logs"; mkdir -p "$LOGS"

GEN=(sbatch "${COMMON[@]}" --array "$ARRAY" --gres "$GRES" --time "$TIME"
     --output "$LOGS/gen_%A_%a.out" "$N4C_ROOT/slurm/generate.sbatch")
echo "models ($NM): ${MODELS[*]}"
echo "tasks: $((LAST + 1)) = $NM models x $NUM_SHARDS shards   results: $RESULTS"
if [ "$DRY" = 1 ]; then
  echo "[dry-run] ${GEN[*]}"
  echo "[dry-run] sbatch ${COMMON[*]} --dependency=afterany:<gen_job_id> --output $LOGS/score_%j.out $N4C_ROOT/slurm/score.sbatch"
  exit 0
fi
GEN_ID=$("${GEN[@]}")
SCORE_ID=$(sbatch "${COMMON[@]}" --dependency="afterany:$GEN_ID" --output "$LOGS/score_%j.out" "$N4C_ROOT/slurm/score.sbatch")
cat <<MSG
submitted: generation array job $GEN_ID, scoring job $SCORE_ID (runs when the array ends)
  watch:    squeue -u \$USER           tail -f $LOGS/gen_${GEN_ID}_0.out
  progress: wc -l $RESULTS/generation/*.jsonl
  results:  $RESULTS/scores/summary.md
  cancel:   scancel $GEN_ID $SCORE_ID
  resume:   re-run the same submit command: finished rows are skipped
MSG
