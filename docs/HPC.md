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

## One-time setup (login node, needs internet)

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
```

Then, on the cluster, export the arguments and submit. **Use exported variables, not `qsub -v` or `sbatch --export`:**
both of those split on commas, and the arguments contain commas.

```bash
export MDFAITH_SEED_DB=$SCRATCH/mdfaith-work/seed.db
export ARGS="reextract --run ff68ec72c737 --extractor ollama:qwen2.5:32b --name pilot-re-extracted-32b"
qsub -V scripts/hpc/job.pbs        # PBS
sbatch scripts/hpc/job.slurm       # SLURM (exports the environment by default)
```

**2. Larger runs on both systems** (the claim extractor defaults to the explainer itself for local models, so pass a
separate one):

```bash
export ARGS="run --system adk_dims,adk_equilibrium --models ollama:qwen2.5:7b --seeds 5 --extractor-model ollama:qwen2.5:32b --name 7b-both-systems"
```
then submit as above.

**3. Image-only**, once a vision model is pulled (names containing `llava`, `vl` or `vision` are treated as vision
models and keep the image-only condition):

```bash
export ARGS="run --system adk_dims --models ollama:<vision-model> --conditions image_only --seeds 5 --extractor-model ollama:qwen2.5:32b --name vision-image-only"
```
then submit as above.

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
