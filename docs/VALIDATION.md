# Validating claim extraction

A model extracts the claims from each explanation, so before I report any result a person has to check that the
extractor read the explanations correctly. This page is the guide for whoever does the labelling. It is designed to take
roughly an hour for 100 claims plus 20 explanations.

## What you are judging (and what you are not)

You are judging **the extractor, not the explanation**. For each claim, decide whether the extracted record faithfully
represents what the explanation says. Whether the statement is *true* is irrelevant here; the verifier decides that
later, and you never see its verdict (the files are blinded).

## Files (in `results/<run>/validation/`)

| File | You edit it? | Purpose |
|---|---|---|
| `claims_to_check.csv` | **yes** | A random sample of extracted claims, stratified by condition |
| `explanations_to_check.csv` | **yes** | A random sample of whole explanations, for the recall check |
| `answer_key.csv` | **no** | The verdicts. Do not open it until you have finished |

## Labelling `claims_to_check.csv`

Fill the `faithful` column for every row:

- **yes**: the sentence in `claim_text` really states what `extracted_as` says (same quantity, same number, same
  residues, same frame).
- **partly**: the number, residues or frame are right but the *quantity or scope* is wrong.
- **no**: the extractor invented or misread something.

Examples of **no** or **partly** that I have already seen from a small extractor:

- A sentence about the *frame-to-frame* RMSD step (0.465 Å) recorded as `rmsd_max`, the trajectory's maximum RMSD.
- A value copied from a table header or a tool name rather than from a claim.
- A suggestion ("it would be useful to compute RMSF") recorded as an assertion.

Use `notes` for anything you want to remember. Leave `faithful` blank only if you cannot decide, and say why.

## Labelling `explanations_to_check.csv`

Read `text`, then enter in `missed_checkable_statements` how many statements the extractor **should** have extracted
but did not (compare with `n_extracted`). A statement counts if it gives a value of RMSD, RMSF, radius of gyration or
hydrogen bonds, ranks residues by flexibility, names a frame for an event, or asserts a cause or mechanism. Enter `0`
if nothing was missed.

## Starting from an AI draft (optional, to save time)

To cut the labelling from about an hour to 20-30 minutes, I drafted every label myself and committed the drafts as
separate files, **not** in the files you label:

- `claims_ai_draft.csv`: my `faithful` label and a one-line reason for each of the 100 claims
- `explanations_ai_draft.csv`: my count of missed checkable statements for each of the 20 explanations

This is a screen, not validation: I am a model judging another model's extraction. Treat it as a first pass you must
overrule wherever you disagree. The policy I used: `yes` when the record matches the sentence; `partly` when the number is
right but the reading is imprecise (a bound recorded as equality, a subset of frames, a nearby quantity); `no` when the
record contradicts the sentence (lowest recorded as max, RMSD recorded as RMSF, a frame-to-frame step recorded as the
maximum RMSD) or the sentence is not a claim (hypotheticals, "data not provided", headings, generic statements).

To load my draft into your files so you can edit in place:

```bash
mdfaith validate prefill --out results/pilot/validation
```

It writes my label into `faithful` (and my reason into `notes`, prefixed `AI draft:`) **only where the cell is blank**; it
never overwrites an entry of yours. **Read every row against its sentence; do not skim.** People shown a draft tend to
accept it (anchoring), so `mdfaith validate score` reports how many of your labels differ from mine and Cohen's kappa
between us. A change rate near zero should make you suspicious rather than reassured. Report the result as
"AI-drafted, human-reviewed", with that change rate.

## Scoring

```bash
mdfaith validate score --out results/pilot/validation
```

It reports extractor precision with a 95% Wilson interval (overall and by claim kind), estimated recall, and the
hallucination rate recomputed using **only the claims you confirmed** next to the rate on all sampled claims.

## How I will use the result

- If precision is at least 0.9 (and at least 100 claims were labelled), I will report results as they are and cite the precision.
- If it is lower, I will report results on confirmed claims only, say how many claims that removed, and re-run with a
  better extractor.
- Either way the labelled files are committed next to the results so anyone can re-score them.
