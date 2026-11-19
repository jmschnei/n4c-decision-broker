#!/usr/bin/env bash
#SBATCH --job-name=n4c-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
# Array job: task i -> model (i mod #models), shard (i div #models). Resource flags here are
# defaults; slurm/submit.sh overrides partition/gres/time/array from the command line.
# Compute nodes do NOT need internet: Ollama, models and data come from the shared WORK dir.
set -euo pipefail

export N4C_ROOT="${N4C_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# Container mode: re-run this very script inside the Pyxis/Enroot image (see config.sh).
if [ -n "${N4C_CONTAINER_IMAGE:-}" ] && [ -z "${N4C_IN_CONTAINER:-}" ]; then
  # shellcheck source=slurm/config.sh
  source "$N4C_ROOT/slurm/config.sh"
  exec srun "${CONTAINER_ARGS[@]}" ${N4C_GRES:+--gres="$N4C_GRES"} --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    --export=ALL,N4C_IN_CONTAINER=1 bash "$N4C_ROOT/slurm/generate.sbatch"
fi
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"

NM=${#MODELS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL="${MODELS[$((IDX % NM))]}"
SHARD=$((IDX / NM))
echo "== host=$(hostname) job=${SLURM_JOB_ID:-local} task=$IDX model=$MODEL shard=$SHARD/$NUM_SHARDS"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "WARNING: no NVIDIA GPU visible (CPU inference will be very slow)"

[ -x "$OLLAMA_BIN" ] || { echo "Ollama not found: build the image (slurm/build_image.sh) or run slurm/setup.sh"; exit 1; }
"$PY" -c "import httpx" 2>/dev/null || { echo "ERROR: $PY ($("$PY" --version 2>&1)) cannot import httpx."; \
  echo "  Use an image built with slurm/build_image.sh (it installs httpx), or delete $VENV and re-run slurm/setup.sh INSIDE the container."; exit 1; }
[ -f "$DATA_NORM/eval_sample.jsonl" ] || { echo "datasets missing: run slurm/setup.sh first"; exit 1; }

# One private Ollama server per task, on a port derived from the job id.
PORT=$((20000 + ${SLURM_JOB_ID:-$RANDOM} % 20000))
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_NUM_PARALLEL="$WORKERS" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1
export OLLAMA_CONTEXT_LENGTH=8192
mkdir -p "$RESULTS/logs" "$RESULTS/generation"
"$OLLAMA_BIN" serve > "$RESULTS/logs/ollama_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 1; done
curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "ollama server did not start"; tail -n 30 "$RESULTS/logs/ollama_${SLURM_JOB_ID:-local}.log"; exit 1; }
curl -fs "http://$OLLAMA_HOST/api/tags" | grep -q "\"name\":\"$MODEL\"" \
  || { echo "model $MODEL not in $OLLAMA_MODELS: run slurm/setup.sh first"; exit 1; }

# Warm-up: load the model into the GPU before timing starts.
curl -fs -m 900 "http://$OLLAMA_HOST/api/chat" -d "{\"model\":\"$MODEL\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":1},\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" >/dev/null \
  || echo "WARNING: warm-up call failed"

EXTRA=()
[ -n "${N4C_LIMIT:-}" ] && EXTRA+=(--limit "$N4C_LIMIT")
"$PY" -I "$N4C_ROOT/tools/eval/generate.py" \
  --model "$MODEL" --data "$DATA_NORM" --out "$RESULTS/generation" \
  --base-url "http://$OLLAMA_HOST" --shots "$SHOTS" --workers "$WORKERS" \
  --shard "$SHARD" --num-shards "$NUM_SHARDS" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== finished $MODEL shard $SHARD"
