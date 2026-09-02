"""Production front matter and tagged audio-file output."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import soundfile as sf
import yaml

from .rendering import RenderResult


SUPPORTED_OUTPUT_TYPES = ("wav", "flac", "mp3", "ogg")


@dataclass(frozen=True, slots=True)
class FrontMatter:
    """Optional, format-neutral metadata authored for one production.

    ``title``, ``series``, ``artist``, ``episode``, and ``season`` map to the
    conventional TITLE/TIT2, ALBUM/TALB, ARTIST/TPE1, TRACKNUMBER/TRCK, and
    DISCNUMBER/TPOS fields in Vorbis comments and ID3v2.4. ``description`` maps
    to the format's description field. ``credits`` is authored display text
    and is combined with minimal Freesound attribution in COMMENT/COMM. FLAC
    and Ogg use Vorbis comments; MP3 output requests ID3v2.4 from ffmpeg.
    """

    series: str | None = None
    episode: int | None = None
    title: str | None = None
    artist: str | None = None
    credits: tuple[str, ...] = ()
    description: str | None = None
    season: int | None = None

    def metadata(self, *, sound_credits: str = "") -> dict[str, str]:
        """Return ffmpeg metadata names common to Vorbis comments and ID3."""

        metadata = {}
        for name in ("title", "artist", "description"):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        if self.series is not None:
            metadata["album"] = self.series
        if self.episode is not None:
            metadata["track"] = str(self.episode)
        if self.season is not None:
            metadata["disc"] = str(self.season)
        comment = credits_comment(self.credits, sound_credits=sound_credits)
        if comment:
            metadata["comment"] = comment
        return metadata


def parse_frontmatter(text: str) -> FrontMatter:
    """Parse and validate the YAML content of a ``<frontmatter>`` element."""

    loaded = yaml.safe_load(text)
    if loaded is None:
        return FrontMatter()
    if not isinstance(loaded, dict):
        raise ValueError("front matter YAML must be a mapping")
    if not all(isinstance(name, str) for name in loaded):
        raise ValueError("front matter field names must be strings")
    allowed = {field.name for field in fields(FrontMatter)}
    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise ValueError(f"unknown front matter field(s): {', '.join(unknown)}")
    for name in ("series", "title", "artist", "description"):
        if (
            name in loaded
            and loaded[name] is not None
            and not isinstance(loaded[name], str)
        ):
            raise ValueError(f"front matter {name} must be a string")
    for name in ("episode", "season"):
        if name in loaded and loaded[name] is not None:
            if not isinstance(loaded[name], int) or isinstance(loaded[name], bool):
                raise ValueError(f"front matter {name} must be an integer")
    credits = loaded.get("credits", ())
    if credits is None:
        credits = ()
    if not isinstance(credits, list) or not all(
        isinstance(item, str) for item in credits
    ):
        raise ValueError("front matter credits must be a list of strings")
    loaded["credits"] = tuple(credits)
    return FrontMatter(**loaded)


def credits_comment(credits: tuple[str, ...], *, sound_credits: str = "") -> str:
    """Build a human-readable Markdown-compatible credits comment."""

    sections = []
    if credits:
        sections.append(
            "## Credits\n\n" + "\n".join(f"- {credit}" for credit in credits)
        )
    if sound_credits.strip():
        sections.append(sound_credits.strip())
    return "\n\n".join(sections)


def write_audio_file(
    path: str | Path,
    result: RenderResult,
    sample_rate: int,
    frontmatter: FrontMatter | None = None,
    *,
    sound_credits: str = "",
) -> None:
    """Encode WAV, FLAC, MP3, or Ogg audio and embed applicable metadata.

    WAV is written directly and does not receive front-matter tags. Other
    formats are encoded by ffmpeg from a temporary float WAV. Ogg Vorbis uses
    the intentionally fixed quality setting 8.5; MP3 uses LAME's quality-based
    VBR mode and ID3v2.4; FLAC uses ffmpeg's native lossless encoder.
    """

    output = Path(path)
    output_type = output.suffix.lower().lstrip(".")
    if output_type not in SUPPORTED_OUTPUT_TYPES:
        raise ValueError(f"Unrecognized output type for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_type == "wav":
        sf.write(output, result.audio, sample_rate)
        return

    metadata = (frontmatter or FrontMatter()).metadata(sound_credits=sound_credits)
    with tempfile.TemporaryDirectory(prefix="radio-drama-output-") as temp_dir:
        source = Path(temp_dir) / "production.wav"
        sf.write(source, result.audio, sample_rate, subtype="FLOAT")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
        ]
        if output_type == "ogg":
            command.extend(["-c:a", "libvorbis", "-q:a", "8.5"])
        elif output_type == "mp3":
            command.extend(["-c:a", "libmp3lame", "-q:a", "2", "-id3v2_version", "4"])
        else:
            command.extend(["-c:a", "flac"])
        for name, value in metadata.items():
            command.extend(["-metadata", f"{name}={value}"])
        command.append(str(output))
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is required for compressed audio output"
            ) from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"ffmpeg audio output failed: {message}") from exc
