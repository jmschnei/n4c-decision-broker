#!/usr/bin/env bash
# Installs Ollama (system-wide, /usr/local) and the Python libraries of the evaluation into the
# image it runs in. Used by docker/Dockerfile.eval and by slurm/build_image.sh (Pyxis
# --container-save). Needs root inside the container.
set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.40.16}"
PIP_PACKAGES="${N4C_PIP_PACKAGES:-httpx rdflib pyoxigraph==0.4.11}"
[ "$(id -u)" = 0 ] || { echo "must run as root inside the container (Pyxis: --container-remap-root)"; exit 1; }

case "$(uname -m)" in x86_64|amd64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;;
  *) echo "unsupported architecture $(uname -m)"; exit 1 ;; esac

# curl + zstd (apt if needed and available; zstd only to unpack the Ollama archive)
need=()
command -v curl >/dev/null 2>&1 || need+=(curl ca-certificates)
command -v zstd >/dev/null 2>&1 || need+=(zstd)
if [ ${#need[@]} -gt 0 ]; then
  command -v apt-get >/dev/null 2>&1 || { echo "missing ${need[*]} and no apt-get"; exit 1; }
  apt-get update -qq && apt-get install -y -qq --no-install-recommends "${need[@]}"
fi

BASE="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION"
PKG="ollama-linux-$ARCH.tar.zst"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR:?}"' EXIT
echo "== downloading $PKG ($OLLAMA_VERSION)"
curl -fsSL --retry 3 -o "$WORKDIR/sha256sum.txt" "$BASE/sha256sum.txt"
curl -fsSL --retry 3 -o "$WORKDIR/$PKG" "$BASE/$PKG"
# sha256sum.txt lists files as "<hash>  ./<name>"
WANT=$(awk -v p="$PKG" '{n=$2; sub(/^\.\//,"",n); if (n==p) print $1}' "$WORKDIR/sha256sum.txt")
GOT=$(sha256sum "$WORKDIR/$PKG" | awk '{print $1}')
[ -n "$WANT" ] && [ "$WANT" = "$GOT" ] || { echo "checksum mismatch for $PKG"; exit 1; }
zstd -dc "$WORKDIR/$PKG" | tar -x -C /usr/local

echo "== python packages: $PIP_PACKAGES"
python3 -m pip install --no-cache-dir -q $PIP_PACKAGES

echo "== verifying"
ollama --version
python3 -c "import httpx, rdflib, pyoxigraph; print('python libs ok, httpx', httpx.__version__)"
printf 'ollama %s\npython %s\nbuilt %s\n' "$OLLAMA_VERSION" "$(python3 --version 2>&1)" "$(date -u +%FT%TZ)" > /etc/n4c-image-info
