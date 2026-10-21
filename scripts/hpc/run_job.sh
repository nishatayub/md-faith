#!/usr/bin/env bash
# Runs ONE mdfaith command against a local Ollama server on the allocated GPU node, then shuts the server down.
# Scheduler-agnostic body: the PBS and SLURM wrappers below call it.
#
# Usage inside a job:   bash scripts/hpc/run_job.sh <mdfaith arguments...>
# Examples:
#   bash scripts/hpc/run_job.sh reextract --run <run-id> --extractor ollama:qwen2.5:32b
#   bash scripts/hpc/run_job.sh run --system adk_dims,adk_equilibrium --models ollama:qwen2.5:7b --seeds 5 --name "7b both systems"
#
# Results go to $MDFAITH_WORK/hpc.db (a separate SQLite file). Copy it back and merge with `mdfaith import-runs`.
set -euo pipefail
source "$(dirname "$0")/env.sh"

DB="${MDFAITH_DB:-$MDFAITH_WORK/hpc.db}"
LOG="$MDFAITH_WORK/ollama-${PBS_JOBID:-${SLURM_JOB_ID:-local}}.log"

# A reextract needs the source run: seed the HPC database from a copy you uploaded, if present.
if [ -n "${MDFAITH_SEED_DB:-}" ] && [ ! -f "$DB" ]; then cp "$MDFAITH_SEED_DB" "$DB"; fi

echo "== starting Ollama on $(hostname) (GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none))"
ollama serve > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null && break; sleep 2; done
curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null || { echo "Ollama did not start; see $LOG"; exit 1; }

cd "$MDFAITH_HOME"
echo "== mdfaith $* --db $DB"
mdfaith "$@" --db "$DB"
echo "== done; results in $DB"
