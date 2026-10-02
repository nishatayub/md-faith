# MD-Faith design

## Question
When an LLM explains a molecular dynamics (MD) analysis, which of its claims are faithful to quantities computed from
the trajectory, and does giving the model tools (instead of a picture or a table) reduce unfaithful claims,
including on trajectories with planted processing problems?

## Positioning (checked 2026-10-02, web search only)
- LLM agents that *run* MD workflows already exist: MDCrow (OpenMM/MDTraj, peer-reviewed), DynaMate (writes text
  interpretations of its plots), NAMD-Agent. Building another agent is not the contribution.
- Generic agent-hallucination benchmarks exist (MIRAGE-Bench, Trajel, SCHEMA); none found for MD explanations scored
  against trajectory-computed ground truth. Re-check before writing up; DynaMate's interpreter is a candidate
  external system to score.

## Conditions
| Condition | Model sees | Purpose |
|---|---|---|
| `image_only` | RMSD / RMSF / contact-map figures | what a vision model does with a plot |
| `table_only` | the same data as downsampled numbers | separates "can't read the plot" from "can't reason" |
| `tool_agent` | analysis tools it chooses to call | does computing beat reading |
| `tool_agent_verified` (planned) | tool agent + a claim-checking pass over its own draft | does self-verification help |

The tool agent never receives ground-truth scalars or the artifact record (tested).

## Ground truth (fixed definitions, `groundtruth.py`)
RMSD (backbone, superposed on frame 0), CA RMSF after fitting, radius of gyration, CA contact occupancy
(8 A, |i-j|>=4), heavy-atom N/O polar-contact occupancy (3.5 A; **not** a full hydrogen-bond criterion, use
MDAnalysis `HydrogenBondAnalysis` with angle criteria before making H-bond claims), RMSD peak frame, RMSD plateau
frame (all later values within 15% of their own mean). RMSD and RMSF are cross-checked against MDAnalysis in tests.

## Claims and verdicts
Explanations are split into atomic claims (`claims.py`):
- **numeric**: a scalar value, checked within a tolerance.
- **ranking**: which residues have the highest/lowest RMSF, checked by overlap with the computed top-k.
- **temporal**: when an RMSD peak or plateau occurs, checked within a frame window.
- **causal**: mechanistic statements. Not checkable from the trajectory; counted as `unsupported_mechanism` unless
  they cite computed evidence, and cited ones go to human review.

Verdicts: `supported`, `contradicted`, `artifact_misread`, `unsupported_mechanism`, `unverifiable`.
`hallucination_rate` = (contradicted + artifact_misread) / checkable claims. Causal claims are reported separately;
they are not folded into the hallucination rate.

## Planted artifacts (`artifacts.py`)
| Artifact | Effect | What a faithful system should do |
|---|---|---|
| `pbc_split` | a domain shifted by one box length in chosen frames: false RMSD spikes | flag as processing problem, not a conformational change |
| `shuffle_frames` | frame order randomised | not make time-ordered claims; notice discontinuities |
| `rigid_jitter` | random rigid motion per frame, trajectory not fitted | handle by fitting; not report raw drift as motion |
Planned: residue-label swap, missing frames, periodic-image jump of a ligand.

## Validity threats (write these into the paper)
1. **Claim extraction is an LLM.** Hand-verify a random subset (target >= 100 claims) and report extractor
   precision/recall; report results with human-checked claims only if the extractor is poor.
2. **Tolerance choices** (10% numeric, 5-frame window, 60% residue overlap) are arbitrary; run sensitivity analysis.
3. **Few systems.** One public system gives narrow conclusions; add 2-3 with different dynamics (folded protein,
   flexible loop, ligand-bound).
4. **Artifacts are synthetic.** State this; where possible add one naturally occurring artifact from a public set.
5. **Prompt sensitivity.** Run >= 3 phrasings of the question; use task-clustered bootstrap (`metrics.py`).
6. **Plot rendering choices** (axes, colormap) affect the image condition; fix and publish them.
7. **Contamination.** AdK is a famous system; models may recite literature. Include at least one less-famous system
   and report whether claims match the literature but not the data.
8. **Provider drift.** Pin model versions and record dates.

## Data rules
Public systems only. Unpublished lab trajectories are not used without the data owner's written permission.
Bundled test data (MDAnalysisTests) is for development; check its licence before redistributing outputs.

## Roadmap
- **Week 1** (this scaffold done): ground truth, artifacts, claim schema, verifier, metrics, offline pipeline. To do: add
  2 more public systems; add `HydrogenBondAnalysis`; build a provider adapter (e.g. Anthropic/OpenAI) that implements
  `LLMClient.complete`.
- **Week 2**: run conditions x models x artifacts x seeds; claim extraction; results tables; hand-verify subset.
- **Week 3**: sensitivity analysis, figures, write-up; optional hidden-state probe on claim-level labels with an
  open-weights model; optional demo page.
