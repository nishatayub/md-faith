#!/usr/bin/env bash
# Run on the cluster login node, from inside the copied kit directory, AFTER loading a Python module that is 3.10 or newer.
# Usage:  module load <python-3.10+ module>   # whichever your cluster provides
#         bash install_offline.sh
# Needs no internet. UNTESTED on any specific cluster.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"

PY="${PYTHON:-python3}"
"$PY" - <<'E'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python {sys.version.split()[0]} is too old: MD-Faith needs 3.10 or newer. Load a newer Python module first.")
print("python", sys.version.split()[0], "ok")
E

: "${MDFAITH_HOME:=$HOME/md-faith}"
: "${MDFAITH_WORK:=${SCRATCH:-$HOME}/mdfaith-work}"
mkdir -p "$MDFAITH_WORK"

echo "== repository -> $MDFAITH_HOME"
mkdir -p "$(dirname "$MDFAITH_HOME")"
tar -xzf "$KIT/repo.tgz" -C "$(dirname "$MDFAITH_HOME")"
[ "$(basename "$MDFAITH_HOME")" = md-faith ] || mv "$(dirname "$MDFAITH_HOME")/md-faith" "$MDFAITH_HOME"

echo "== virtual environment (offline)"
cd "$MDFAITH_HOME"
"$PY" -m venv .venv
.venv/bin/pip install --no-index --find-links "$KIT/wheelhouse" --upgrade pip setuptools wheel || true
.venv/bin/pip install --no-index --find-links "$KIT/wheelhouse" \
  "MDAnalysis>=2.7" MDAnalysisData numpy scipy pandas matplotlib pyyaml pytest
# source-only package: only its data files are used, so skip its dependencies and its build isolation (no network)
.venv/bin/pip install --no-index --no-deps --no-build-isolation --find-links "$KIT/wheelhouse" MDAnalysisTests
.venv/bin/pip install --no-index --no-deps -e .

echo "== Ollama"
OLLAMA_HOME="${OLLAMA_HOME:-$MDFAITH_WORK/ollama}"
mkdir -p "$OLLAMA_HOME"
tar -xzf "$KIT/ollama-linux-amd64.tgz" -C "$OLLAMA_HOME"
"$OLLAMA_HOME/bin/ollama" --version

echo
echo "Done. Next: copy model weights into \$MDFAITH_WORK/ollama-models (see docs/HPC.md), then source scripts/hpc/env.sh"
.venv/bin/mdfaith --help > /dev/null && echo "mdfaith CLI imports fine"
