from radio_drama.text import normalize_text_punctuation


def test_normalize_text_punctuation_uses_ascii_equivalents():
    assert normalize_text_punctuation(
        "‘single’ ‚low‛ “double” „low‟ wait–what—really…"
    ) == """'single' 'low' "double" "low" wait--what---really..."""


def test_normalize_text_punctuation_preserves_existing_ascii_text():
    text = """'single' "double" wait--what---really..."""

    assert normalize_text_punctuation(text) == text
