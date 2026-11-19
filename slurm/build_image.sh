#!/usr/bin/env bash
# Build the evaluation image on the cluster with Pyxis (no Docker needed): start the base image,
# install Ollama + Python libs inside it, and save the result as a .sqsh in your folder.
#
#   bash slurm/build_image.sh --base /enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
#        --out /netscratch/$USER/n4c-eval.sqsh [--partition P] [--time HH:MM:SS] [--mounts "/ds:/ds:ro"]
#
# Needs internet in the container (downloads Ollama from GitHub) and Pyxis' --container-save and
# --container-remap-root (root inside the container, required to write to /usr/local).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="" OUT="" PARTITION="" TIME="01:00:00" MOUNTS="" OLLAMA_VERSION="${OLLAMA_VERSION:-v0.40.16}"
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2 ;;  --out) OUT=$2; shift 2 ;;  --partition) PARTITION=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;  --mounts) MOUNTS=$2; shift 2 ;;
    --ollama-version) OLLAMA_VERSION=$2; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
[ -n "$BASE" ] && [ -n "$OUT" ] || { echo "--base and --out are required"; exit 2; }
[ -f "$BASE" ] || { echo "base image not found: $BASE"; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT already exists: remove or choose another --out"; exit 1; }
mkdir -p "$(dirname "$OUT")"

echo "building $OUT from $BASE (Ollama $OLLAMA_VERSION)"
srun ${PARTITION:+--partition="$PARTITION"} --time="$TIME" --cpus-per-task=4 --mem=16G \
  --container-image="$BASE" --container-save="$OUT" --container-remap-root \
  --container-workdir="$ROOT" --container-mounts="$ROOT:$ROOT${MOUNTS:+,$MOUNTS}" \
  --export=ALL,OLLAMA_VERSION="$OLLAMA_VERSION" \
  bash "$ROOT/docker/install_in_image.sh"
ls -lh "$OUT"
echo "done. Use it with:  bash slurm/submit.sh --container-image $OUT ..."
