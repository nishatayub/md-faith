#!/usr/bin/env bash
# One-time setup on a LOGIN node (needs internet): Python environment, optional trajectory data, Ollama, model weights.
# Usage:  source scripts/hpc/env.sh && bash scripts/hpc/setup.sh qwen2.5:7b qwen2.5:32b
# Pass the model names you want as arguments. Compute nodes often have no internet, so pull models here, not in the job.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== Python environment"
cd "$MDFAITH_HOME"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[systems]"

echo "== Optional second system (about 160 MB)"
.venv/bin/mdfaith fetch-data adk_equilibrium || echo "skipped: could not fetch adk_equilibrium"

echo "== Ollama (user-space install, no root needed)"
if [ ! -x "$OLLAMA_HOME/bin/ollama" ]; then
  mkdir -p "$OLLAMA_HOME"
  # Official Linux tarball; check https://github.com/ollama/ollama/blob/main/docs/linux.md if this URL has moved.
  curl -fL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C "$OLLAMA_HOME"
fi
ollama --version

echo "== Model weights"
ollama serve > "$MDFAITH_WORK/ollama-setup.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null && break; sleep 2; done
for m in "$@"; do ollama pull "$m"; done
ollama list
