from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from carthage.dependency_injection import AsyncInjector, Injector

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.effects import EffectChainRegistry
from radio_drama.init import radio_drama_injector
from radio_drama.planning import PRODUCTION_PLANNING_INJECTOR_KEY


_TEST_SPEAKER_LINE_RE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$")


@dataclass(slots=True)
class PlaceholderAudioPlan:
    node: object | None = None

    def __repr__(self) -> str:
        return "PlaceholderAudioPlan()"


async def make_async_injector(
    config: ProductionConfig,
    *,
    document_path: Path | None = None,
    output_path: Path | None = None,
    effect_chains: EffectChainRegistry | None = None,
) -> tuple[Injector, AsyncInjector]:
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
        document_path=document_path,
        output_path=output_path,
    )
    if effect_chains is None:
        effect_chains = EffectChainRegistry()
        # Most unit tests exercise planning and composition rather than the
        # external ffmpeg mastering integration.  Keep that boundary explicit
        # and deterministic; mastering tests construct the built-in registry.
        effect_chains.add_from_expression("master", "dry()")
    injector.add_provider(effect_chains)
    injector.add_provider(PRODUCTION_PLANNING_INJECTOR_KEY, injector)
    return injector, injector(AsyncInjector)


def request_from_normalized_script(
    normalized_script: str,
    voice_samples: tuple[str, ...],
    *,
    first_words: str = "",
) -> ScriptRenderRequest:
    speaker_refs: dict[int, SpeakerVoiceReference] = {}
    dialogue_lines: list[DialogueLine] = []
    for raw_line in normalized_script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TEST_SPEAKER_LINE_RE.match(line)
        assert match is not None, line
        speaker_number = int(match.group(1))
        voice_sample = voice_samples[speaker_number - 1]
        speaker_refs.setdefault(
            speaker_number,
            SpeakerVoiceReference(
                authored_name=f"Speaker {speaker_number}",
                voice_name=Path(voice_sample).name,
                resolved_path=Path(voice_sample),
            ),
        )
        dialogue_lines.append(
            DialogueLine(
                speaker=speaker_refs[speaker_number],
                spoken_text=match.group(2).strip(),
            )
        )
    return ScriptRenderRequest(dialogue_lines=dialogue_lines, first_words=first_words)


def normalized_script_from_request(request: ScriptRenderRequest) -> str:
    speaker_numbers: dict[str, int] = {}
    normalized_lines: list[str] = []
    for line in request.dialogue_lines:
        speaker_key = line.speaker.authored_name.lower()
        speaker_number = speaker_numbers.get(speaker_key)
        if speaker_number is None:
            speaker_number = len(speaker_numbers) + 1
            speaker_numbers[speaker_key] = speaker_number
        normalized_lines.append(f"Speaker {speaker_number}: {line.spoken_text}")
    return "\n".join(normalized_lines).replace("’", "'")
