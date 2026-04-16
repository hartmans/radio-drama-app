from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radio_drama.forced_alignment import (
    WhisperXResponse,
    _alignment_result_from_whisperx_response,
    fill_start_positions_from_alignment,
)
from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference


_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s*(.*)$")


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def _parse_dialogue_text(text: str | None) -> list[tuple[str, str]]:
    text = re.sub(r"^\s*\n", "", text or "")
    text = re.sub(r"\n\s*$", "", text)
    if not text:
        return []

    lines: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_paragraph: list[str] = []
    current_paragraphs: list[str] = []

    def flush_paragraph() -> None:
        if current_paragraph:
            current_paragraphs.append(" ".join(current_paragraph).strip())
            current_paragraph.clear()

    def flush_stanza() -> None:
        nonlocal current_speaker
        flush_paragraph()
        if current_speaker is None:
            return
        spoken_text = " ".join(paragraph for paragraph in current_paragraphs if paragraph).strip()
        current_paragraphs.clear()
        if spoken_text:
            lines.append((current_speaker, spoken_text))

    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            flush_paragraph()
            continue
        match = _SPEAKER_LINE_RE.match(stripped_line)
        if match is not None:
            flush_stanza()
            current_speaker = match.group(1).strip()
            current_paragraph.append(match.group(2).strip())
            continue
        if current_speaker is None:
            raise ValueError(f"Scripts may begin only with a speaker stanza, got: {raw_line!r}")
        current_paragraph.append(stripped_line)

    flush_stanza()
    return lines


def _extract_script_lines(xml_path: Path, script_index: int) -> list[tuple[str, str]]:
    root = ET.parse(xml_path).getroot()
    scripts = root.findall(".//script")
    try:
        script = scripts[script_index]
    except IndexError as exc:
        raise ValueError(f"Script index {script_index} out of range for {xml_path}") from exc

    lines: list[tuple[str, str]] = []
    if script.text:
        lines.extend(_parse_dialogue_text(script.text))
    for child in script:
        if child.tag == "line":
            lines.append((child.attrib["speaker"], _normalize_text("".join(child.itertext()))))
        elif child.tag in {"group", "ignore"}:
            lines.extend(_parse_dialogue_text("".join(child.itertext())))
        if child.tail:
            lines.extend(_parse_dialogue_text(child.tail))
    return lines


def _expected_dialogue_lines(
    transcript_lines: list[tuple[str, str]],
    response: WhisperXResponse,
    *,
    transcript: str,
    duration_seconds: float,
) -> list[dict[str, object]]:
    speakers: dict[str, SpeakerVoiceReference] = {}
    contents: list[DialogueLine] = []
    for speaker_name, spoken_text in transcript_lines:
        speakers.setdefault(
            speaker_name,
            SpeakerVoiceReference(
                authored_name=speaker_name,
                voice_name=f"{speaker_name}.wav",
                resolved_path=Path(f"{speaker_name}.wav"),
            ),
        )
        contents.append(
            DialogueLine(
                speaker=speakers[speaker_name],
                spoken_text=spoken_text,
            )
        )

    alignment = _alignment_result_from_whisperx_response(
        transcript,
        response,
        duration_seconds=duration_seconds,
    )
    filled = fill_start_positions_from_alignment(contents, alignment)
    return [
        {
            "speaker": speaker_name,
            "spoken_text": spoken_text,
            "expected_start_seconds": (
                None if math.isnan(content.start_pos) else content.start_pos
            ),
        }
        for (speaker_name, spoken_text), content in zip(transcript_lines, filled, strict=True)
    ]


def build_case(
    *,
    whisperx_json: Path,
    xml_path: Path,
    script_index: int,
    name: str,
) -> dict[str, object]:
    payload = json.loads(whisperx_json.read_text(encoding="utf-8"))
    response = WhisperXResponse(
        transcription_segments=tuple(payload["transcription_segments"]),
        aligned_segments=(
            None
            if payload["aligned_segments"] is None
            else tuple(payload["aligned_segments"])
        ),
        decision=payload["decision"],
    )
    duration_seconds = max(
        (
            float(segment.get("end", 0.0) or 0.0)
            for segment in payload["transcription_segments"]
        ),
        default=0.0,
    )
    transcript_lines = _extract_script_lines(xml_path, script_index)
    return {
        "name": name,
        "transcript": payload["transcript"],
        "duration_seconds": duration_seconds,
        "whisperx_response": {
            "decision": payload["decision"],
            "transcription_segments": payload["transcription_segments"],
            "aligned_segments": payload["aligned_segments"],
        },
        "dialogue_lines": _expected_dialogue_lines(
            transcript_lines,
            response,
            transcript=payload["transcript"],
            duration_seconds=duration_seconds,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("whisperx_json", type=Path)
    parser.add_argument("xml_path", type=Path)
    parser.add_argument("script_index", type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    case_payload = build_case(
        whisperx_json=args.whisperx_json,
        xml_path=args.xml_path,
        script_index=args.script_index,
        name=args.name,
    )
    output_text = json.dumps(case_payload, indent=2) + "\n"
    if args.output is None:
        print(output_text, end="")
        return 0
    args.output.write_text(output_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
