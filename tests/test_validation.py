import csv

import pytest

from mdfaith.claims import Claim, Kind, Label, Verdict
from mdfaith.store import Store
from mdfaith.validation import make_sample, score, wilson


@pytest.fixture()
def store_with_claims():
    s = Store()
    rid = s.create_run("r")
    for i in range(30):
        cond = ["table_only", "tool_agent", "qc_gated"][i % 3]
        c = Claim(
            "c1", f"The mean backbone RMSD is {i}.0 Å.", Kind.NUMERIC, {"quantity": "rmsd_mean", "value": float(i)}
        )
        lab = Label.SUPPORTED if i % 4 == 0 else Label.CONTRADICTED
        s.add_explanation(rid, "adk_dims:none", "none", cond, "m", f"text {i}", [c], [Verdict("c1", lab, "d")], seed=i)
    return s, rid


def test_sample_is_blinded_stratified_and_deterministic(store_with_claims, tmp_path):
    s, rid = store_with_claims
    res = make_sample(s, rid, tmp_path, n_claims=12, n_explanations=5, seed=1)
    assert res["claims_sampled"] == 12 and res["explanations_sampled"] == 5
    rows = list(csv.DictReader(open(tmp_path / "claims_to_check.csv")))
    assert "label" not in rows[0] and "detail" not in rows[0]  # blinded: no verdicts in the annotation file
    assert {r["condition"] for r in rows} == {"table_only", "tool_agent", "qc_gated"}
    assert rows[0]["extracted_as"].startswith("rmsd_mean = ")
    key = list(csv.DictReader(open(tmp_path / "answer_key.csv")))
    assert {r["claim_row_id"] for r in key} == {r["claim_row_id"] for r in rows}
    a = [r["claim_row_id"] for r in rows]
    make_sample(s, rid, tmp_path / "again", n_claims=12, n_explanations=5, seed=1)
    assert a == [r["claim_row_id"] for r in csv.DictReader(open(tmp_path / "again" / "claims_to_check.csv"))]


def test_wilson_interval():
    p, lo, hi = wilson(90, 100)
    assert p == 0.9 and 0.82 < lo < 0.85 and 0.94 < hi < 0.96
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN for empty input


def test_score_precision_recall_and_confirmed_rate(store_with_claims, tmp_path):
    s, rid = store_with_claims
    make_sample(s, rid, tmp_path, n_claims=30, n_explanations=4, seed=0)
    rows = list(csv.DictReader(open(tmp_path / "claims_to_check.csv")))
    for i, r in enumerate(rows):  # mark 3 in 4 faithful, one in 4 unfaithful, a couple partly, a couple blank
        r["faithful"] = ["yes", "yes", "yes", "no"][i % 4]
    rows[0]["faithful"] = "partly"
    rows[1]["faithful"] = ""
    with open(tmp_path / "claims_to_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ex = list(csv.DictReader(open(tmp_path / "explanations_to_check.csv")))
    for r in ex:
        r["missed_checkable_statements"] = "1"
    with open(tmp_path / "explanations_to_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ex[0]))
        w.writeheader()
        w.writerows(ex)
    res = score(tmp_path / "claims_to_check.csv", tmp_path / "answer_key.csv", tmp_path / "explanations_to_check.csv")
    assert res["claims_annotated"] == 29 and res["claims_unlabelled"] == 1
    prec = res["extractor_precision"]
    assert prec["faithful"] + prec["partly"] + prec["unfaithful"] == 29 and 0 < prec["value"] < 1
    assert prec["ci95"][0] < prec["value"] < prec["ci95"][1]
    assert res["rate_on_confirmed_claims"]["checkable"] <= res["rate_on_all_sampled_claims"]["checkable"]
    assert res["enough_for_a_claim"] is False  # fewer than 100 annotated
    assert res["estimated_recall"]["missed"] == 4 and 0 < res["estimated_recall"]["value"] < 1


def test_unknown_run_rejected(tmp_path):
    with pytest.raises(KeyError):
        make_sample(Store(), "nope", tmp_path)


def test_draft_agreement_kappa_and_confusion(tmp_path):
    from mdfaith.validation import draft_agreement

    claims = tmp_path / "c.csv"
    draft = tmp_path / "d.csv"
    final = ["yes", "yes", "no", "no", "partly", "yes", "no", "partly"]
    ai = ["yes", "yes", "no", "yes", "partly", "no", "no", "no"]
    with open(claims, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_row_id", "faithful"])
        for i, v in enumerate(final):
            w.writerow([i, v])
    with open(draft, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_row_id", "ai_draft_faithful", "ai_draft_reason"])
        for i, v in enumerate(ai):
            w.writerow([i, v, ""])
    res = draft_agreement(claims, draft)
    assert res["n"] == 8 and res["agreement"] == pytest.approx(5 / 8) and res["changed"] == 3
    assert -1 <= res["cohens_kappa"] <= 1 and res["confusion_draft_by_human"]["yes"]["no"] == 1
    assert draft_agreement(claims, draft)["confusion_draft_by_human"]["no"]["partly"] == 1


def test_identical_draft_and_final_gives_kappa_one(tmp_path):
    from mdfaith.validation import draft_agreement

    labels = ["yes", "no", "partly", "yes", "no"]
    for name, col in (("c.csv", "faithful"), ("d.csv", "ai_draft_faithful")):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["claim_row_id", col])
            for i, v in enumerate(labels):
                w.writerow([i, v])
    res = draft_agreement(tmp_path / "c.csv", tmp_path / "d.csv")
    assert res["agreement"] == 1.0 and res["changed"] == 0 and res["cohens_kappa"] == pytest.approx(1.0)


def test_prefill_fills_blanks_only_and_never_overwrites_a_human(tmp_path):
    from mdfaith.validation import prefill

    with open(tmp_path / "claims_to_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_row_id", "condition", "artifact", "kind", "claim_text", "extracted_as", "faithful", "notes"])
        w.writerow([1, "c", "a", "numeric", "t", "x", "", ""])
        w.writerow([2, "c", "a", "numeric", "t", "x", "no", "mine"])
    with open(tmp_path / "claims_ai_draft.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_row_id", "ai_draft_faithful", "ai_draft_reason"])
        w.writerow([1, "yes", "matches"])
        w.writerow([2, "yes", "matches"])
    with open(tmp_path / "explanations_to_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["explanation_id", "condition", "artifact", "n_extracted", "text", "missed_checkable_statements", "notes"]
        )
        w.writerow([9, "c", "a", 2, "text", "", ""])
    with open(tmp_path / "explanations_ai_draft.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["explanation_id", "ai_draft_missed", "ai_draft_reason"])
        w.writerow([9, 3, "counted"])
    res = prefill(tmp_path)
    assert res == {"claims_prefilled": 1, "explanations_prefilled": 1}
    rows = list(csv.DictReader(open(tmp_path / "claims_to_check.csv")))
    assert rows[0]["faithful"] == "yes" and rows[0]["notes"].startswith("AI draft:")
    assert rows[1]["faithful"] == "no" and rows[1]["notes"] == "mine"  # the human's entry is untouched
    assert next(csv.DictReader(open(tmp_path / "explanations_to_check.csv")))["missed_checkable_statements"] == "3"
    assert prefill(tmp_path) == {"claims_prefilled": 0, "explanations_prefilled": 0}  # idempotent
