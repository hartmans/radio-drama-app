from pathlib import Path
from types import SimpleNamespace

from radio_drama.freesound import (
    FreesoundCredit,
    likely_freesound_ids,
    load_api_key,
    markdown_credits,
)
from radio_drama.sound import SoundPlan, sound_plans_in


class PlanGroup:
    def __init__(self, children):
        self.children = children

    def all_plans(self):
        yield self
        for child in self.children:
            if isinstance(child, PlanGroup):
                yield from child.all_plans()
            else:
                yield child


def sound(ref: str, resolved_path: str | None = None) -> SoundPlan:
    plan = object.__new__(SoundPlan)
    plan.node = SimpleNamespace(ref=ref)
    plan.resolved_path = Path(resolved_path) if resolved_path is not None else None
    return plan


def test_sound_plans_in_extracts_and_deduplicates_assets():
    first = sound("foley/gavel", "/sounds/762733__science_witch__gavel.wav")
    duplicate = sound(
        "762733__science_witch__gavel",
        "/sounds/762733__science_witch__gavel.wav",
    )
    unresolved = sound("ambience/rain")
    repeated_unresolved = sound("ambience/rain")
    root = PlanGroup([first, PlanGroup([duplicate, unresolved]), repeated_unresolved])

    assert sound_plans_in(root) == (first, unresolved)


def test_likely_freesound_ids_uses_initial_number_and_deduplicates():
    root = PlanGroup(
        [
            sound("762733__science_witch__gavel.wav"),
            sound("library/12345_wind.flac"),
            sound("not-a-freesound.wav"),
            sound("762733_duplicate.mp3"),
        ]
    )

    assert likely_freesound_ids(root) == (762733, 12345)


def test_load_api_key_reads_secret_from_yaml(tmp_path, monkeypatch):
    credentials = tmp_path / "freesound.yml"
    credentials.write_text("client_id: client-value\nsecret: token-value\n")
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)

    assert load_api_key(credentials) == "token-value"


def test_minimal_markdown_is_plain_text_readable():
    credit = FreesoundCredit(123, "A Bell", "Sound Author")

    assert markdown_credits((credit,), minimal=True) == (
        "## Sound credits\n\n"
        "- “A Bell” by Sound Author — https://freesound.org/s/123/\n"
    )
