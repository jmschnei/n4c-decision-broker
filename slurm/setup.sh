#!/usr/bin/env bash
# One-time setup: Python venv, Ollama (user-space, no root), models, datasets.
# Needs internet: run it on the login node (bash slurm/setup.sh) or via
# `sbatch slurm/setup.sbatch` on a partition with internet access. Idempotent: re-run it freely.
#
# Skip parts with SKIP_VENV=1 SKIP_OLLAMA=1 SKIP_MODELS=1 SKIP_DATA=1.
set -euo pipefail

export N4C_ROOT="${N4C_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=slurm/config.sh
source "$N4C_ROOT/slurm/config.sh"
PYTHON="${PYTHON:-python3}"
mkdir -p "$WORK"
echo "== N4C_ROOT=$N4C_ROOT  WORK=$WORK  OLLAMA=$OLLAMA_VERSION"

free_gb() { df -Pk "$1" | awk 'NR==2{printf "%d", $4/1048576}'; }
echo "free space in WORK: $(free_gb "$WORK") GB (need ~100 GB for the 7 models + Ollama)"

# ---------------------------------------------------------------- 1. Python venv
if [ -z "${SKIP_VENV:-}" ] && "$PYTHON" -c "import httpx, rdflib" 2>/dev/null; then
  echo "== python libs already available in this environment (image): skipping venv"
elif [ -z "${SKIP_VENV:-}" ]; then
  if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c "import httpx" 2>/dev/null; then
    echo "   existing venv is unusable here (created elsewhere?): recreating"; rm -rf "${VENV:?}"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV" 2>/dev/null || {  # some container images lack python3-venv
      echo "   python -m venv failed, falling back to virtualenv"
      rm -rf "${VENV:?}"
      "$PYTHON" -m pip install -q --user virtualenv && "$PYTHON" -m virtualenv -q "$VENV"
    }
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -e "$N4C_ROOT[eval]" zstandard
  echo "== venv ready: $("$VENV/bin/python" --version)"
fi
# pyoxigraph pinned: ORDER BY ... LIMIT ties are broken by the engine's internal order, so another
# version can change exec_match. Images without it (or with another version) get it in $WORK/pylib,
# which score.py puts first on sys.path (N4C_PYLIB).
PYX="$(pick_python || echo "$PYTHON")"
OXV=0.4.11
if [ -z "${SKIP_VENV:-}" ] \
   && ! PYTHONPATH="$WORK/pylib" "$PYX" -c "import pyoxigraph as o; assert o.__version__ == '$OXV'" 2>/dev/null; then
  echo "== installing pyoxigraph $OXV into $WORK/pylib (local Spider4SPARQL execution)"
  "$PYX" -m pip install -q --upgrade --target "$WORK/pylib" "pyoxigraph==$OXV" \
    || echo "   WARNING: pyoxigraph not installed; Spider4SPARQL will be scored offline only"
fi

# ---------------------------------------------------------------- 2. Ollama (no root)
if [ -z "${SKIP_OLLAMA:-}" ] && [ ! -x "$OLLAMA_BIN" ] && command -v ollama >/dev/null 2>&1; then
  echo "== using ollama from the image: $(command -v ollama) ($(ollama --version 2>&1 | tail -1))"
elif [ -z "${SKIP_OLLAMA:-}" ]; then
  if [ -x "$OLLAMA_BIN" ] && "$OLLAMA_BIN" --version 2>&1 | grep -q "${OLLAMA_VERSION#v}"; then
    echo "== ollama $OLLAMA_VERSION already installed"
  else
    case "$(uname -m)" in x86_64|amd64) ARCH=amd64 ;; aarch64|arm64) ARCH=arm64 ;;
      *) echo "unsupported architecture $(uname -m)"; exit 1 ;; esac
    BASE="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION"
    DL="$WORK/downloads"; mkdir -p "$DL" "$OLLAMA_DIR"
    curl -fsSL --retry 3 -o "$DL/sha256sum.txt" "$BASE/sha256sum.txt"
    PKGS=("ollama-linux-$ARCH.tar.zst")
    [ "$OLLAMA_FLAVOR" = "rocm" ] && PKGS+=("ollama-linux-$ARCH-rocm.tar.zst")
    for PKG in "${PKGS[@]}"; do
      echo "== downloading $PKG (~1.5 GB)"
      curl -fL --retry 3 -C - -o "$DL/$PKG" "$BASE/$PKG"
      WANT=$(awk -v p="$PKG" '{n=$2; sub(/^\.\//,"",n); if (n==p) print $1}' "$DL/sha256sum.txt")
      GOT=$( (sha256sum "$DL/$PKG" 2>/dev/null || shasum -a 256 "$DL/$PKG") | awk '{print $1}')
      [ -n "$WANT" ] && [ "$WANT" = "$GOT" ] || { echo "checksum mismatch for $PKG"; exit 1; }
      if command -v zstd >/dev/null 2>&1; then zstd -dc "$DL/$PKG" | tar -x -C "$OLLAMA_DIR"
      else "$VENV/bin/python" -c "import sys,zstandard;zstandard.ZstdDecompressor().copy_stream(open(sys.argv[1],'rb'),sys.stdout.buffer)" "$DL/$PKG" | tar -x -C "$OLLAMA_DIR"
      fi
      rm -f "${DL:?}/${PKG:?}"
    done
    OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
    echo "== installed: $("$OLLAMA_BIN" --version 2>&1 | tail -1)"
  fi
fi

# ---------------------------------------------------------------- 3. Models
if [ -z "${SKIP_MODELS:-}" ]; then
  mkdir -p "$OLLAMA_MODELS"
  PORT=$((20000 + RANDOM % 20000))
  export OLLAMA_HOST="127.0.0.1:$PORT"
  "$OLLAMA_BIN" serve > "$WORK/ollama_pull_server.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 1 60); do curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 1; done
  curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "ollama server did not start"; tail "$WORK/ollama_pull_server.log"; exit 1; }
  for M in "${MODELS[@]}"; do
    echo "== pulling $M"
    for TRY in 1 2 3; do "$OLLAMA_BIN" pull "$M" && break || { echo "retry $TRY"; sleep 10; }; done
  done
  "$OLLAMA_BIN" list
  MISSING=0
  for M in "${MODELS[@]}"; do
    "$OLLAMA_BIN" list | awk 'NR>1{print $1}' | grep -qx "$M" || { echo "MISSING: $M"; MISSING=1; }
  done
  [ "$MISSING" = 0 ] || exit 1
  kill $SERVER 2>/dev/null || true
fi

# ---------------------------------------------------------------- 4. Datasets
if [ -z "${SKIP_DATA:-}" ]; then
  # Third-party data: one directory per dataset, parsed only (never executed).
  fetch() { # name url dest-filename
    mkdir -p "$DATA_RAW/$1"
    curl -fsSL --retry 3 -o "$DATA_RAW/$1/$3" "$2" && echo "   $1/$3 $(wc -c < "$DATA_RAW/$1/$3") bytes"
  }
  echo "== downloading datasets"
  fetch spider4sparql https://raw.githubusercontent.com/ckosten/Spider4SPARQL/HEAD/nl_sparql_pairs/dev/dev.json dev.json
  fetch spider4sparql https://raw.githubusercontent.com/ckosten/Spider4SPARQL/HEAD/LICENSE LICENSE
  # Materialized dev KGs (~10 MB zip on Google Drive, linked from the Spider4SPARQL README):
  # used by score.py --spider-kg to execute Spider queries locally. Only the .ttl files are kept.
  # Known defects: orchestra.ttl is empty, wta_1.ttl is truncated (those items get excluded).
  if [ ! -f "$DATA_RAW/spider4sparql_kg/concert_singer.ttl" ]; then
    mkdir -p "$DATA_RAW/spider4sparql_kg_zip" "$DATA_RAW/spider4sparql_kg"
    if curl -fsSL --retry 3 -o "$DATA_RAW/spider4sparql_kg_zip/dev_kgs.zip" \
         "https://drive.google.com/uc?export=download&id=1S6xaI0VfnFMrsuyjxv2uQPok_CikhLLx" \
       && unzip -tq "$DATA_RAW/spider4sparql_kg_zip/dev_kgs.zip" >/dev/null 2>&1; then
      unzip -qo -j "$DATA_RAW/spider4sparql_kg_zip/dev_kgs.zip" 'materialized_triples/*.ttl' -d "$DATA_RAW/spider4sparql_kg"
      echo "   spider4sparql_kg: $(ls "$DATA_RAW/spider4sparql_kg" | wc -l) KGs"
    else
      echo "   WARNING: could not download the Spider4SPARQL KGs (Google Drive); Spider is scored offline only"
    fi
  fi
  for F in instantiated_data paraphrased_data template_questions; do
    fetch golem "https://raw.githubusercontent.com/GOLEM-lab/GOLEM_Text-to-SPARQL/HEAD/dataset/$F.tsv" "$F.tsv"
  done
  fetch qald10 https://raw.githubusercontent.com/KGQA/QALD-10/HEAD/data/qald_10/qald_10.json qald_10.json
  fetch qald10 https://raw.githubusercontent.com/KGQA/QALD-10/HEAD/LICENSE LICENSE
  fetch qald9plus https://raw.githubusercontent.com/KGQA/QALD-10/HEAD/data/qald_9_plus/qald_9_plus_train_wikidata.json qald_9_plus_train_wikidata.json
  echo "== normalizing"
  PY="$PYTHON"; "$VENV/bin/python" -c pass 2>/dev/null && PY="$VENV/bin/python"
  "$PY" -I "$N4C_ROOT/tools/normalize_datasets.py" "$DATA_RAW" "$DATA_NORM" | tail -n 12
fi
echo "== setup finished"
