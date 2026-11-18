# Shared configuration for the Slurm evaluation (sourced by the other scripts).
# Override any value with an environment variable before calling setup.sh / submit.sh.
#
#   N4C_WORK   big shared filesystem directory (scratch/project space), ~100 GB free:
#              Ollama binary, model blobs (~80 GB), datasets, venv, results.

ROOT="${N4C_ROOT:?N4C_ROOT must point to the repository root}"
WORK="${N4C_WORK:-$ROOT/cluster_work}"
RUN_NAME="${N4C_RUN_NAME:-eval}"

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.40.16}"   # version tested while developing (nimble needs >= 0.40)
OLLAMA_FLAVOR="${OLLAMA_FLAVOR:-}"             # "" = NVIDIA/CPU build; "rocm" for AMD GPUs
OLLAMA_DIR="$WORK/ollama"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$WORK/ollama_models}"

# Ollama keeps its identity key in $HOME/.ollama: make sure HOME is writable (containers often aren't).
if [ ! -w "${HOME:-/nonexistent}" ]; then export HOME="$WORK/home"; mkdir -p "$HOME"; fi

VENV="$WORK/venv"
DATA_RAW="$WORK/datasets/raw"
DATA_NORM="$WORK/datasets/normalized"
RESULTS="$WORK/results/$RUN_NAME"

# Candidate models (see README: evaluation). Override: N4C_MODELS="qwen3-coder:30b llama3.1:8b"
if [ -n "${N4C_MODELS:-}" ]; then
  read -r -a MODELS <<< "$N4C_MODELS"
else
  MODELS=(
    "qwen3-coder:30b"
    "qwen2.5-coder:latest"
    "qwen3.8:27b"
    "gemma4:12b"
    "llama3.1:8b"
    "glm-4.7-flash:latest"
    "llama3.2:3b"
  )
fi

NUM_SHARDS="${N4C_SHARDS:-2}"     # every model is split over this many parallel GPU jobs
WORKERS="${N4C_WORKERS:-4}"       # concurrent requests per Ollama server (OLLAMA_NUM_PARALLEL)
SHOTS="${N4C_SHOTS:-3}"

# Optional Pyxis/Enroot container (e.g. N4C_CONTAINER_IMAGE=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh).
# When set, the sbatch scripts re-run themselves inside it with `srun --container-image ...`.
# Extra mounts (comma separated, e.g. "/ds:/ds:ro") go in N4C_CONTAINER_MOUNTS.
CONTAINER_ARGS=()
if [ -n "${N4C_CONTAINER_IMAGE:-}" ]; then
  CONTAINER_ARGS=(
    "--container-image=$N4C_CONTAINER_IMAGE"
    "--container-workdir=$ROOT"
    "--container-mounts=$WORK:$WORK,$ROOT:$ROOT${N4C_CONTAINER_MOUNTS:+,$N4C_CONTAINER_MOUNTS}"
  )
fi
