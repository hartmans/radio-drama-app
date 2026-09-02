import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from radio_drama.cache import cache_directory_for_output
from radio_drama.cli import initialize_arg_parser, resolved_output_path
from radio_drama.document import parse_production_string
from radio_drama.frontmatter import FrontMatter, parse_frontmatter, write_audio_file
from radio_drama.rendering import RenderResult


def test_frontmatter_element_parses_all_optional_fields():
    document = parse_production_string(
        """
        <production>
          <frontmatter>
            series: Nothing but the Succubus
            episode: 3
            title: Mysterious Mister X
            artist: Sam Hartman
            credits:
              - Qwen TTS Voice Design
              - MOSS Voice Design
            description: A mysterious encounter.
            season: 1
          </frontmatter>
        </production>
        """
    )

    assert document.frontmatter == FrontMatter(
        series="Nothing but the Succubus",
        episode=3,
        title="Mysterious Mister X",
        artist="Sam Hartman",
        credits=("Qwen TTS Voice Design", "MOSS Voice Design"),
        description="A mysterious encounter.",
        season=1,
    )


def test_empty_production_has_empty_frontmatter():
    assert parse_production_string("<production />").frontmatter == FrontMatter()


@pytest.mark.parametrize("suffix", ["flac", "mp3", "ogg"])
def test_write_audio_file_encodes_audio_and_metadata(tmp_path, suffix):
    output = tmp_path / f"episode.{suffix}"
    result = RenderResult(audio=np.zeros((480, 2), dtype=np.float32))
    frontmatter = parse_frontmatter(
        """
        series: Example Series
        episode: 4
        title: Example Episode
        artist: Example Artist
        credits:
          - Qwen TTS Voice Design
        description: Example description
        season: 2
        """
    )

    write_audio_file(
        output,
        result,
        48_000,
        frontmatter,
        sound_credits="## Sound credits\n\n- “Bell” by Example — https://example.test/bell",
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags:stream_tags",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    raw_tags = payload["format"].get("tags", {})
    for stream in payload.get("streams", ()):
        raw_tags.update(stream.get("tags", {}))
    tags = {key.lower(): value for key, value in raw_tags.items()}
    assert tags["title"] == "Example Episode"
    assert tags["album"] == "Example Series"
    assert tags["artist"] == "Example Artist"
    assert tags["track"] == "4"
    assert tags["disc"] == "2"
    assert "Qwen TTS Voice Design" in tags["comment"]
    assert "“Bell” by Example" in tags["comment"]


def test_cache_directory_always_uses_wav_name():
    assert cache_directory_for_output("episode.mp3").name == "episode.wav.cache"
    assert cache_directory_for_output("episode.ogg").name == "episode.wav.cache"
    assert cache_directory_for_output("episode.flac").name == "episode.wav.cache"


def test_output_type_selects_suffix_and_is_exclusive_with_output():
    parser = initialize_arg_parser("test")
    args = parser.parse_args(["episode.xml", "--output-type", "flac"])
    assert resolved_output_path(args) == Path("episode.flac")

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["episode.xml", "--output", "named.mp3", "--output-type", "ogg"]
        )


def test_output_rejects_unrecognized_suffix():
    parser = initialize_arg_parser("test")
    with pytest.raises(SystemExit):
        parser.parse_args(["episode.xml", "--output", "episode.aac"])
