"""TextGrid parsing + the tag policy (keep <UNSURE> text, drop inaudible)."""

from s2n.data.textgrid import clean_text, count_tags


def test_unsure_content_is_kept_tags_dropped():
    assert clean_text("<UNSURE>Hello how</UNSURE> are you") == "Hello how are you"


def test_inaudible_markers_are_stripped():
    assert clean_text("Okay <UNIN/> sorry") == "Okay sorry"
    assert clean_text("<INAUDIBLE_SPEECH/>") == ""


def test_whitespace_is_collapsed():
    assert clean_text("  a   b  ") == "a b"


def test_tag_counts():
    tc = count_tags("<UNSURE>x</UNSURE> <UNIN/> <UNIN/> <INAUDIBLE_SPEECH/>")
    assert tc.unsure == 1
    assert tc.inaudible == 3  # 2x UNIN + 1x INAUDIBLE_SPEECH
