# Contributing to MD-Faith

Thanks for helping. The project is small and opinionated; these rules keep it trustworthy.

## Ground rules
1. **Public data only.** Never add unpublished trajectories without the data owner's written permission.
2. **No LLM judges for ground truth.** If a claim can be checked against the trajectory, check it with code.
3. **Demo data must be labelled.** Anything simulated is flagged `is_demo` and shown as such in the UI and figures.
4. **Every behaviour change has a test.** Ground-truth definitions are cross-checked against MDAnalysis where possible.

## A tip for macOS
Keep the repository outside iCloud-synced folders such as Desktop and Documents. Two things went wrong for me there:
iCloud created duplicate files named like `name 2.py`, which ended up in a commit, and macOS later **evicted** about a
hundred tracked files (including objects inside `.git`) as "dataless" files. With sync unavailable, any read of an
evicted file blocks forever, so Python imports, `pytest` and even `git` hung with no error. You can spot it with
`ls -lO`: the flags column shows `compressed,dataless`. A fresh clone in a normal folder such as `~/code` fixed it.

## Workflow
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
pre-commit install            # optional
make check                    # lint + tests
```
- One feature per branch, one branch per pull request, named `feat/Fxx-short-name`.
- Use the PR template. Keep PRs reviewable (< ~600 changed lines where possible).
- Commit messages: imperative mood, say *why* in the body when it is not obvious.

## Adding a trajectory artifact
Implement a builder in `src/mdfaith/artifacts.py` returning `(universe, ArtifactRecord)`, register it in `BUILDERS`,
state in the docstring what a faithful system should do, and add a test showing the ground truth changes as claimed.

## Adding a claim type
Extend `claims.Kind`, add a verifier in `verify.py` with explicit tolerances, update `extract.py`'s prompt and the
rule-based extractor, and document the definition in `docs/DESIGN.md`.
