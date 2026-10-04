import pytest

from mdfaith import artifacts, groundtruth
from mdfaith.data import load_system
from mdfaith.extract_rules import RuleExtractor, split_sentences
from mdfaith.report import annotate, summary, to_markdown
from mdfaith.verify import verify_all


@pytest.fixture(scope="module")
def u():
    return load_system("adk_dims")


@pytest.fixture(scope="module")
def gt(u):
    return groundtruth.compute(u, "adk_dims")


def test_split_sentences_keeps_decimals_together():
    s = split_sentences(
        "Mean backbone RMSD is 4.4 A. RMSD peaks at frame 90, so it moves. Residues 1 and 2 are flexible."
    )
    assert len(s) == 3 and s[0].endswith("4.4 A.")


def test_rule_extractor_patterns():
    text = (
        "The mean backbone RMSD is 4.4 Å. Residues 149, 150 and 151 show the highest RMSF. "
        "RMSD peaks at frame 90, indicating a large conformational change. "
        "The radius of gyration is 18.3 Å. There are about 110 hydrogen bonds per frame."
    )
    cs = RuleExtractor().extract(text)
    kinds = [(c.kind.value, c.payload.get("quantity") or c.payload.get("event")) for c in cs]
    assert ("numeric", "rmsd_mean") in kinds and ("numeric", "rg_mean") in kinds
    assert ("numeric", "hbond_mean_count") in kinds and ("ranking", "rmsf") in kinds
    assert ("temporal", "rmsd_peak") in kinds and any(k == "causal" for k, _ in kinds)
    rank = next(c for c in cs if c.kind.value == "ranking")
    assert rank.payload["residues"] == [149, 150, 151] and rank.payload["direction"] == "highest"
    assert all("_sentence" in c.payload for c in cs)


def test_artifact_language_marks_event_as_not_physical():
    cs = RuleExtractor().extract("RMSD peaks at frame 30, which is a wrapping artifact.")
    t = next(c for c in cs if c.kind.value == "temporal")
    assert t.payload["physical"] is False
    assert not any(c.kind.value == "causal" for c in cs)


def test_annotate_faithful_vs_unfaithful_report(gt):
    s = gt.scalars
    good = f"The mean backbone RMSD is {s['rmsd_mean']:.1f} Å. RMSD peaks at frame {s['rmsd_max_frame']}."
    bad = "The mean backbone RMSD is 12.0 Å. Residues 1, 2 and 3 show the highest RMSF."
    for text, expect in [(good, {"supported"}), (bad, {"contradicted"})]:
        cs = RuleExtractor().extract(text)
        ann = annotate(text, cs, verify_all(cs, gt), gt)
        assert {a.status for a in ann} == expect
    ev = ann[0].evidence[0]
    assert (
        ev.computed == pytest.approx(s["rmsd_mean"], abs=1e-3)
        and ev.tool == "rmsd"
        and "superposition" in ev.definition
    )


def test_annotate_flags_artifact_misread_with_description(u):
    u2, rec = artifacts.build_default(u, "pbc_split")
    g2 = groundtruth.compute(u2)
    text = f"RMSD peaks at frame {rec.frames[0]}, showing a large conformational change."
    cs = RuleExtractor().extract(text)
    ann = annotate(text, cs, verify_all(cs, g2, rec), g2, rec)
    assert ann[0].status == "artifact"
    assert any("shifted" in (e.definition or "") for e in ann[0].evidence)
    assert summary(ann)["artifact"] == 1


def test_unchecked_sentences_and_markdown(gt):
    text = "The protein looks fine. The mean backbone RMSD is 1.0 Å."
    cs = RuleExtractor().extract(text)
    ann = annotate(text, cs, verify_all(cs, gt), gt)
    assert [a.status for a in ann] == ["unchecked", "contradicted"]
    md = to_markdown(ann)
    assert md.splitlines()[0].startswith("[-]") and "[WRONG]" in md


def test_alignment_falls_back_to_token_overlap(gt):
    from mdfaith.claims import Claim, Kind

    text = "The structure is stable. The radius of gyration stays near 18 Å throughout."
    c = Claim("x", "radius of gyration stays near 18", Kind.NUMERIC, {"quantity": "rg_mean", "value": 18.3})
    ann = annotate(text, [c], verify_all([c], gt), gt)
    assert ann[1].evidence and ann[0].status == "unchecked"
