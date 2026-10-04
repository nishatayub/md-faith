# Findings to fill in

Placeholders exist so nothing gets forgotten. Each row names where it lives, what replaces it, and what blocks it.
Nothing marked **placeholder** or **demo** is a result.

| # | Where | Currently | Replace with | Blocked by |
|---|---|---|---|---|
| 1 | Landing site, *Findings*, Figure 1 | Dashed placeholder | `docs/figures/fig1_hallucination_by_condition.png` from a **real-model** run | Live-API run (needs credentials) |
| 2 | Landing site, *Findings*, Figure 2 | Dashed placeholder | `fig3_artifact_misread_share.png` from the real run | same |
| 3 | Landing site, *Findings*, Figure 3 | Dashed placeholder | Hidden-state probe result (stretch goal) | Open-weights model + claim-level labels |
| 4 | README, results table | Demo numbers, labelled | Measured numbers with 95% intervals | Real-model run |
| 5 | `docs/figures/*` (fig1-3, tables) | Stamped DEMO from simulated explainers | Regenerate with `mdfaith figures --run <real run>` | Real-model run |
| 6 | Poster, results panel | Placeholder boxes | Same figures as above | Real-model run |
| 7 | Methods | Claim-extractor validity | Precision / recall of the LLM extractor on >= 100 hand-checked claims | Hand annotation |
| 8 | Methods | Tolerance sensitivity | Sweep of numeric tolerance, frame window and residue overlap | Real-model run |
| 9 | Systems | One system (AdK) | Two or more further public systems, at least one not famous (contamination check) | Choosing and downloading public data |
| 10 | API adapter | Tested with a fake SDK client only | Live smoke test of `AnthropicClient` | API credentials |

Unchanged real artifact: `fig4_artifact_rmsd.png` is computed ground truth (no model involved) and can stay.
