"""Shared text transformations for model and analysis boundaries."""

_ASCII_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2013": "--",
        "\u2014": "---",
        "\u2026": "...",
    }
)


def normalize_text_punctuation(text: str) -> str:
    """Return text with common typographic punctuation replaced by ASCII forms.

    Single and double typographic quotation marks become ``'`` and ``"``,
    en and em dashes become ``--`` and ``---``, and the ellipsis character
    becomes ``...``. Other text is preserved.
    """

    return text.translate(_ASCII_PUNCTUATION_TRANSLATION)


__all__ = ["normalize_text_punctuation"]
