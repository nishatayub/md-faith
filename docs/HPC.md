# Running the heavy parts on a GPU cluster

My laptop has 8.6 GB of RAM, which is not enough to run a 7B model comfortably (it left about 80 MB free and the machine
hung). Everything that needs a bigger model therefore runs on the IIT Delhi HPC GPUs. This page is the recipe.

> **Status: untested on IIT Delhi.** I wrote these scripts from the documented behaviour of Ollama and standard PBS and
> SLURM, without access to the cluster. Expect to adjust queue names, resource lines and module loads. Everything marked
> `CHANGE_ME` must be filled in. The merge step (`mdfaith import-runs`) is covered by tests.

## What needs a GPU

| Open item | What to run | Why it cannot run on the laptop |
|---|---|---|
| Re-extract the pilot with a separate, stronger extractor | `reextract` with a 32B model | 7B already exhausts the laptop; 32B needs about 20 GB of GPU memory |
| Larger experiments on both systems | `run` with 7B and 32B explainers | Many long generations; 7B is too slow and too large locally |
| Image-only condition | `run` with a vision model | Needs a vision-language model of useful size |

## How it works

The code needs no changes. A job starts `ollama serve` on the GPU node, runs one `mdfaith` command against it, and
stops the server. Results go into a **separate SQLite file** on the cluster, which you copy back and merge into your
local database, so the two never overwrite each other.

```
laptop                                  cluster
------                                  -------
git clone / git pull   ------------->   $MDFAITH_HOME           (repo)
results/mdfaith.db     --scp-------->   $MDFAITH_WORK/seed.db   (only for reextract)
                                        qsub / sbatch job  ->  $MDFAITH_WORK/hpc.db
results/hpc.db         <--scp--------   $MDFAITH_WORK/hpc.db
mdfaith import-runs --from results/hpc.db
```

## If the login node has no internet (the IIT Delhi case)

On IIT Delhi's login nodes, `git clone` and `pip` time out (no internet, no proxy set), and the system Python is 3.9 while
MD-Faith needs 3.10 or newer. So build an **offline kit** on a machine with internet (your laptop), copy it over, and
install from it. The scheduler there is PBS.

**1. On your laptop**, choose the cluster Python minor version first. Find a module that is 3.10 or newer on the login node
(`module avail 2>&1 | grep -i -E "anaconda|python"`, then `module load <it>` and `python3 --version`). Then:

```bash
bash scripts/hpc/make_offline_kit.sh 3.11
```

This writes `mdfaith-offline-kit/` with a snapshot of the repository, Python wheels built for linux x86_64 (the cluster is
`x86_64`) and the Ollama server. The wheel download needs the Python version you pass to match the module you load.

```bash
rsync -avP mdfaith-offline-kit/ YOUR_LOGIN:\$SCRATCH/mdfaith-kit/
```

**2. Model weights, also from your laptop.** Pull the model locally, copy the weights, then delete them locally if disk is
tight (Ollama stores weights by content hash, so copying the whole folder merges safely):

```bash
ollama pull qwen2.5:14b
```

```bash
rsync -avP ~/.ollama/models/ YOUR_LOGIN:\$SCRATCH/mdfaith-work/ollama-models/
```

Sizes matter because a laptop with limited disk cannot hold a 32B model (about 20 GB). Practical choices are `qwen2.5:7b`
(about 4.7 GB) as the explainer and `qwen2.5:14b` (about 9 GB) as a separate, larger extractor. Check the exact names
and sizes on the [Ollama library](https://ollama.com/library). For image-only, pull one vision model the same way.

**3. On the cluster login node:**

```bash
module load <the python module you chose>
```

```bash
cd $SCRATCH/mdfaith-kit && PYTHON=python3 bash install_offline.sh
```

```bash
cd ~/md-faith && source scripts/hpc/env.sh && ollama list
```

`ollama list` should show the models you copied. Then continue with the jobs below. If the job nodes have no GPU modules
loaded by default, add the `module load` line your cluster uses for CUDA to the job script.

## One-time setup (login node WITH internet only)

```bash
git clone https://github.com/nishatayub/md-faith.git ~/md-faith
cd ~/md-faith
export MDFAITH_WORK=$SCRATCH/mdfaith-work        # or any large work/scratch path
source scripts/hpc/env.sh
bash scripts/hpc/setup.sh qwen2.5:7b qwen2.5:32b # model names to pull; add a vision model for image-only
```

`setup.sh` creates the virtual environment, optionally fetches the second system (about 160 MB), installs Ollama in
user space (no root) and pulls the models. Pull models on the login node: compute nodes often have no internet.
Check the model names against the [Ollama library](https://ollama.com/library) before pulling.

## Jobs

Use the PBS or SLURM template, whichever your cluster runs, after filling in the queue, resources and walltime.

**1. Re-extract the pilot with a separate, stronger extractor** (the model that wrote the explanations should not also
extract their claims):

```bash
scp results/mdfaith.db cluster:$SCRATCH/mdfaith-work/seed.db
# PBS:
qsub -v MDFAITH_SEED_DB=$SCRATCH/mdfaith-work/seed.db,ARGS="reextract --run ff68ec72c737 --extractor ollama:qwen2.5:32b --name pilot-re-extracted-32b" scripts/hpc/job.pbs
# SLURM:
sbatch --export=ALL,MDFAITH_SEED_DB=$SCRATCH/mdfaith-work/seed.db,ARGS="reextract --run ff68ec72c737 --extractor ollama:qwen2.5:32b --name pilot-re-extracted-32b" scripts/hpc/job.slurm
```

**2. Larger runs on both systems** (the claim extractor defaults to the explainer itself for local models, so pass a
separate one):

```bash
ARGS="run --system adk_dims,adk_equilibrium --models ollama:qwen2.5:7b --seeds 5 --extractor-model ollama:qwen2.5:32b --name 7b-both-systems"
```

**3. Image-only**, once a vision model is pulled (names containing `llava`, `vl` or `vision` are treated as vision
models and keep the image-only condition):

```bash
ARGS="run --system adk_dims --models ollama:<vision-model> --conditions image_only --seeds 5 --extractor-model ollama:qwen2.5:32b --name vision-image-only"
```

## Bringing results back

```bash
scp cluster:$SCRATCH/mdfaith-work/hpc.db results/hpc.db
mdfaith import-runs --from results/hpc.db            # copies every finished run into results/mdfaith.db
mdfaith compare-extractions --a ff68ec72c737 --b <re-extracted-run-id>
mdfaith export --run <run-id> --out results/<name>
mdfaith figures --run <run-id> --out results/<name>
```

`compare-extractions` reports how often two extractors agree on the claims in the same explanations. It is a **proxy**
for extraction quality, not a validation: two extractors can agree and both be wrong. The human check in
[VALIDATION.md](VALIDATION.md) is still required before any result is reported.

## Things to check on the cluster

- Queue or partition name, GPU request syntax, walltime limits, and whether a `module load cuda` is needed.
- Whether compute nodes can reach the internet (the scripts assume they cannot, so models are pulled in advance).
- Storage quotas: a 32B model is about 20 GB, a 70B model about 40 GB.
- Whether running a long-lived local server inside a job is allowed on your cluster's usage policy.
