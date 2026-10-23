#!/usr/bin/env bash
# Shared settings for the MD-Faith HPC scripts. Edit the three paths, then `source scripts/hpc/env.sh`.
# UNTESTED on any specific cluster: written from the documented behaviour of Ollama and standard PBS.

# Where things live. Use scratch or work storage, not a small home quota: model files are many GB.
export MDFAITH_HOME="${MDFAITH_HOME:-$HOME/md-faith}"                 # the cloned repository
export MDFAITH_WORK="${MDFAITH_WORK:-${SCRATCH:-$HOME}/mdfaith-work}"  # databases, logs, models
export OLLAMA_HOME="${OLLAMA_HOME:-$MDFAITH_WORK/ollama}"              # the unpacked Ollama binary

export OLLAMA_MODELS="$MDFAITH_WORK/ollama-models"   # model weights
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="2h"
export PATH="$OLLAMA_HOME/bin:$MDFAITH_HOME/.venv/bin:$PATH"

mkdir -p "$MDFAITH_WORK" "$OLLAMA_MODELS"
