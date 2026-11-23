"""Unit tests of the snippet retrieval and the rule baselines (no data or model needed)."""
from periop.notes.snippets import classify_anes_phrase, retrieve, rule_anes, rule_ebl


def test_ebl_number_with_unit():
    assert rule_ebl("ESTIMATED BLOOD LOSS: 200 mL.") == "200"
    assert rule_ebl("Estimated blood loss was approximately 150 cc.") == "150"
    assert rule_ebl("BLOOD LOSS: 1.5 L") == "1500"
    assert rule_ebl("EBL: 1,200 ml") == "1200"


def test_ebl_less_than_keeps_the_number():
    assert rule_ebl("ESTIMATED BLOOD LOSS: Less than 50 cc.") == "50"


def test_ebl_words():
    assert rule_ebl("ESTIMATED BLOOD LOSS: Minimal.") == "minimal"
    assert rule_ebl("EBL: Scant.") == "minimal"


def test_ebl_absent():
    assert rule_ebl("The patient tolerated the procedure well.") == "not stated"


def test_ebl_ignores_other_volumes():
    # the fluid volume belongs to another header and must not be read as blood loss
    assert rule_ebl("ESTIMATED BLOOD LOSS: Minimal.\nFLUIDS: 800 mL of crystalloid.") == "minimal"


def test_anaesthesia_classes():
    assert classify_anes_phrase("General endotracheal") == "general"
    assert classify_anes_phrase("Spinal with sedation") == "regional"
    assert classify_anes_phrase("General and spinal") == "combined"
    assert classify_anes_phrase("MAC") == "local_or_sedation"
    assert classify_anes_phrase("Local with 1% lidocaine") == "local_or_sedation"


def test_anaesthesia_header():
    assert rule_anes("PROCEDURE: Hernia repair.\nANESTHESIA: General.\n") == "general"
    assert rule_anes("No header in this note.") == "not stated"


def test_retrieval_window_contains_the_value():
    text = "x" * 500 + " ESTIMATED BLOOD LOSS: 300 mL. " + "y" * 500
    snips = retrieve(text, "ebl")
    assert len(snips) == 1 and "300 mL" in snips[0].text and len(snips[0].text) < 400


def test_retrieval_empty_when_no_mention():
    assert retrieve("Nothing relevant here.", "ebl") == []
