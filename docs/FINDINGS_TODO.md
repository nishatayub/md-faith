# Findings to fill in

Placeholders exist so nothing gets forgotten. Each row names where it lives, what replaces it, and what blocks it.
Nothing marked **placeholder** or **demo** is a result.

| # | Where | Currently | Replace with | Blocked by |
|---|---|---|---|---|
| 1 | Landing site, *Findings*, Figure 1 | Dashed placeholder | `docs/figures/fig1_hallucination_by_condition.png` from a **real-model** run | Live-API run (needs credentials) |
| 2 | Landing site, *Findings*, Figure 2 | Dashed placeholder | `fig3_artifact_misread_share.png` from the real run | same |
| 3 | Landing site, *Findings*, Figure 3 | Dashed placeholder | Hidden-state probe result (stretch goal) | Open-weights model + claim-level labels |
| 4 | README, results table | Pilot (Qwen2.5-3B, 3 conditions) plus labelled demo numbers | Larger run with a separate extractor and hand-validated claims | Stronger model access, hand annotation |
| 5 | `docs/figures/*` (fig1-3, tables) | Stamped DEMO from simulated explainers | Regenerate with `mdfaith figures --run <real run>` | Real-model run |
| 6 | Poster, results panel | Placeholder boxes | Same figures as above | Real-model run |
| 7 | Methods | Claim-extractor validity: workflow and blinded sample built, **not yet labelled** | Precision / recall of the extractor on >= 100 hand-checked claims (`mdfaith validate score`) | A person labelling `results/pilot/validation/` (see `docs/VALIDATION.md`) |
| 8 | Methods | Tolerance sensitivity | Sweep of numeric tolerance, frame window and residue overlap | Real-model run |
| 9 | Systems | Two systems, **both AdK** (bundled transition run and a 1 us equilibrium run) | At least one non-famous protein for the contamination check (e.g. other MDAnalysisData sets); needs a short-run-compatible system with hydrogens | Choosing and approving the download |
| 10 | API adapter | Tested with a fake SDK client only | Live smoke test of `AnthropicClient` | API credentials |

Unchanged real artifact: `fig4_artifact_rmsd.png` is computed ground truth (no model involved) and can stay.
