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
