#!/usr/bin/env bash
# Run on a machine WITH internet (your laptop). Builds a folder you copy to a cluster whose login nodes have no internet.
#
# Usage:  bash scripts/hpc/make_offline_kit.sh <cluster-python-minor> [output-dir]
#   <cluster-python-minor>  the Python the cluster will use, e.g. 3.11 or 3.12 (must be 3.10 or newer)
#
# The kit contains:  repo.tgz (this repository at HEAD), wheelhouse/ (Python packages built for linux x86_64 and that
# Python version), and ollama-linux-amd64.tgz (the Ollama server). Model weights are copied separately; see docs/HPC.md.
set -euo pipefail

PYV="${1:?give the cluster Python version, e.g. 3.11}"
OUT="${2:-mdfaith-offline-kit}"
case "$PYV" in 3.1[0-9]) ;; *) echo "Python must be 3.10 or newer (got $PYV)"; exit 1;; esac
ABI="cp${PYV/./}"

mkdir -p "$OUT/wheelhouse"
echo "== repository snapshot"
git archive --format=tar.gz --prefix=md-faith/ -o "$OUT/repo.tgz" HEAD

echo "== wheelhouse for linux x86_64, Python $PYV (binary wheels only)"
python3 -m pip download -d "$OUT/wheelhouse" \
  --only-binary=:all: --implementation cp --python-version "$PYV" --abi "$ABI" \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --platform any \
  "MDAnalysis>=2.7" MDAnalysisData numpy scipy pandas matplotlib pyyaml pytest setuptools wheel pip

# MDAnalysisTests (the bundled public trajectory) is published only as a source archive, so fetch it on its own.
python3 -m pip download -d "$OUT/wheelhouse" --no-deps --no-binary=:all: MDAnalysisTests

echo "== Ollama server (about 1.5 GB)"
if [ ! -f "$OUT/ollama-linux-amd64.tgz" ]; then
  curl -fL -o "$OUT/ollama-linux-amd64.tgz" https://ollama.com/download/ollama-linux-amd64.tgz
fi

cp scripts/hpc/install_offline.sh "$OUT/install_offline.sh"
echo
echo "Kit ready in $OUT ($(du -sh "$OUT" | cut -f1)). Copy it to the cluster, e.g.:"
echo "  rsync -avP $OUT/ YOUR_LOGIN:\$SCRATCH/mdfaith-kit/"
