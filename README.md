# MD-Faith

Do LLM explanations of molecular dynamics analyses stay faithful to what the trajectory actually shows?

MD-Faith is an **evaluation harness**, not another MD agent (MDCrow, DynaMate and others already run MD workflows).
It computes ground truth from a trajectory, plants known processing problems (periodic-boundary wrapping, shuffled
frames, unfitted frames), asks LLMs to explain the analysis under different conditions (plot only, table only,
tool-using agent), splits explanations into atomic claims, and scores each claim programmatically.

Status: scaffold. Core ground truth, artifacts, verifier, metrics and offline pipeline are implemented and tested
(20 tests). No LLM provider adapter is included yet and no results have been produced.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 20 tests, a few seconds
mdfaith selftest               # offline end-to-end check with scripted fake model replies
mdfaith groundtruth --system adk_dims --out results/gt_adk.json
```

## Layout
```
src/mdfaith/
  data.py          public systems
  groundtruth.py   RMSD, RMSF, Rg, contact and polar-contact occupancy, event frames
  artifacts.py     pbc_split, shuffle_frames, rigid_jitter (+ records of affected frames)
  claims.py        claim kinds, verdict labels, JSON schema
  verify.py        programmatic claim verification (no LLM judge)
  metrics.py       hallucination rate, task-clustered bootstrap CIs
  llm.py           LLMClient protocol + ScriptedClient for offline tests
  render.py        plots (image-only) and tables (table-only)
  tools.py         tools exposed to the agent condition
  conditions/      image_only, table_only, tool_agent
  extract.py       LLM claim extraction prompt + validation
  pipeline.py      build tasks, score an explanation
  cli.py
configs/default.yaml
docs/DESIGN.md     conditions, claim taxonomy, validity threats, roadmap
tests/
```

## What is still needed
1. An `LLMClient` adapter for the model providers you will test (`llm.py`); keys come from the environment, never the repo.
2. More public systems and a hydrogen-bond analysis with angle criteria.
3. The experiment runner (conditions x models x artifacts x seeds) and results tables.
4. Hand verification of a random subset of extracted claims.

See `docs/DESIGN.md` for the validity threats that must be addressed in any write-up.

## Data
Public systems only. Do not add unpublished trajectories without written permission from the data owner.
