from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml
from carthage.dependency_injection import AsyncInjectable, Injector, inject

from .config import ProductionConfig
from .document import DocumentNode, ProductionNode, ScriptNode, SpeakerMapNode
from .errors import DocumentError
from .rendering import ProductionResult, RenderResult


_SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}
_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s*(.*)$")


@dataclass(frozen=True, slots=True)
class SpeakerVoiceReference:
    authored_name: str
    voice_name: str
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class DialogueLine:
    speaker: SpeakerVoiceReference
    spoken_text: str


@dataclass(frozen=True, slots=True)
class ScriptRenderRequest:
    normalized_script: str
    voice_samples: tuple[str, ...]


@inject(injector=Injector)
class PlanningNode(AsyncInjectable):
    def __init__(self, node: DocumentNode | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node = node
        self._render_task: asyncio.Task | None = None

    def document_error(self, message: str, *, node: DocumentNode | None = None) -> DocumentError:
        target = node or self.node
        if target is None:
            return DocumentError(message)
        return target.error(message)

    async def render(self):
        if self._render_task is None:
            self._render_task = asyncio.create_task(self.render_node())
        try:
            return await self._render_task
        except BaseException:
            self._render_task = None
            raise

    async def render_node(self):
        return None


@inject(config=ProductionConfig)
class SpeakerMapPlan(PlanningNode):
    def __init__(self, node: SpeakerMapNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self._voices_by_key: dict[str, SpeakerVoiceReference] = {}

    async def async_ready(self):
        loaded = yaml.safe_load(self.node.normalized_text_content)
        if not isinstance(loaded, dict):
            raise self.document_error(
                "The <speaker-map> YAML must be a mapping of speaker names to voice names"
            )
        if not loaded:
            raise self.document_error("The <speaker-map> did not define any speakers")

        voices_by_key: dict[str, SpeakerVoiceReference] = {}
        for speaker_name, voice_name in loaded.items():
            if not isinstance(speaker_name, str) or not isinstance(voice_name, str):
                raise self.document_error(
                    "Speaker names and voice names in <speaker-map> must be strings"
                )
            normalized_speaker = speaker_name.strip()
            normalized_voice = voice_name.strip()
            if not normalized_speaker or not normalized_voice:
                raise self.document_error(
                    "Speaker names and voice names in <speaker-map> cannot be empty"
                )
            key = normalized_speaker.lower()
            if key in voices_by_key:
                raise self.document_error(
                    f"Speaker {speaker_name!r} is defined more than once in <speaker-map>"
                )
            voices_by_key[key] = SpeakerVoiceReference(
                authored_name=normalized_speaker,
                voice_name=normalized_voice,
                resolved_path=self._resolve_voice_path(normalized_speaker, normalized_voice),
            )

        self._voices_by_key = voices_by_key
        return await super().async_ready()

    def lookup(self, speaker_name: str) -> SpeakerVoiceReference:
        return self._voices_by_key[speaker_name.strip().lower()]

    @property
    def voices_by_key(self) -> Mapping[str, SpeakerVoiceReference]:
        return self._voices_by_key

    def _resolve_voice_path(self, speaker_name: str, voice_name: str) -> Path:
        direct_candidates = [
            Path(voice_name).expanduser(),
            (self.config.resolved_voice_directory / voice_name).expanduser(),
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        voice_catalog = self._load_voice_catalog()
        candidate_keys = [
            voice_name,
            Path(voice_name).name,
            Path(voice_name).stem,
            voice_name.lower(),
            Path(voice_name).name.lower(),
            Path(voice_name).stem.lower(),
        ]
        for candidate in candidate_keys:
            resolved = voice_catalog.get(candidate)
            if resolved is not None:
                return resolved

        available = ", ".join(sorted({path.name for path in voice_catalog.values()}))
        raise self.document_error(
            f"Voice {voice_name!r} for speaker {speaker_name!r} was not found in "
            f"{self.config.resolved_voice_directory}. Available voices: {available}"
        )

    def _load_voice_catalog(self) -> dict[str, Path]:
        voice_directory = self.config.resolved_voice_directory
        if not voice_directory.is_dir():
            raise self.document_error(f"Voice directory does not exist: {voice_directory}")

        catalog: dict[str, Path] = {}
        for child in sorted(voice_directory.iterdir()):
            if not child.is_file() or child.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS:
                continue
            catalog.setdefault(child.name, child)
            catalog.setdefault(child.stem, child)
            catalog.setdefault(child.name.lower(), child)
            catalog.setdefault(child.stem.lower(), child)

        if not catalog:
            raise self.document_error(f"No supported voice files were found in {voice_directory}")
        return catalog

    async def render_node(self):
        return None


@inject(speaker_map_plan=SpeakerMapPlan)
class ScriptPlan(PlanningNode):
    def __init__(self, node: ScriptNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.dialogue_lines: list[DialogueLine] = []
        self.ordered_speakers: list[SpeakerVoiceReference] = []
        self.render_request: ScriptRenderRequest | None = None
        self._registered_request = None

    async def async_ready(self):
        self.dialogue_lines = self._parse_dialogue()
        self.ordered_speakers = self._ordered_unique_speakers(self.dialogue_lines)
        if len(self.ordered_speakers) > 4:
            raise self.document_error(
                f"A <script> uses {len(self.ordered_speakers)} speakers, but VibeVoice supports at most 4"
            )
        if self.dialogue_lines:
            local_speaker_ids = {
                speaker.authored_name.lower(): index + 1
                for index, speaker in enumerate(self.ordered_speakers)
            }
            normalized_script = "\n".join(
                f"Speaker {local_speaker_ids[line.speaker.authored_name.lower()]}: {line.spoken_text}"
                for line in self.dialogue_lines
            ).replace("’", "'")
            self.render_request = ScriptRenderRequest(
                normalized_script=normalized_script,
                voice_samples=tuple(str(ref.resolved_path) for ref in self.ordered_speakers),
            )
        from .resources import VibeVoiceResource

        resource = await self.ainjector.get_instance_async(VibeVoiceResource)
        self._registered_request = await resource.register_request(self.render_request)
        return await super().async_ready()

    async def render_node(self) -> RenderResult:
        return await self._registered_request.render()

    def _parse_dialogue(self) -> list[DialogueLine]:
        text = self.node.normalized_text_content
        if not text:
            return []

        lines: list[DialogueLine] = []
        current_speaker: SpeakerVoiceReference | None = None
        current_paragraph: list[str] = []
        current_paragraphs: list[str] = []

        def flush_paragraph() -> None:
            if current_paragraph:
                current_paragraphs.append(" ".join(current_paragraph).strip())
                current_paragraph.clear()

        def flush_stanza() -> None:
            flush_paragraph()
            if current_speaker is None:
                return
            spoken_text = " ".join(paragraph for paragraph in current_paragraphs if paragraph).strip()
            current_paragraphs.clear()
            if spoken_text:
                lines.append(DialogueLine(speaker=current_speaker, spoken_text=spoken_text))

        for raw_line in text.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line:
                flush_paragraph()
                continue
            match = _SPEAKER_LINE_RE.match(stripped_line)
            if match is not None:
                candidate_speaker = match.group(1).strip()
                try:
                    speaker_ref = self.speaker_map_plan.lookup(candidate_speaker)
                except KeyError:
                    speaker_ref = None
                if speaker_ref is not None:
                    flush_stanza()
                    current_speaker = speaker_ref
                    current_paragraph.append(match.group(2).strip())
                    continue
            if current_speaker is None:
                raise self.document_error(
                    "Scripts may begin only with a recognized `speaker:` stanza"
                )
            current_paragraph.append(stripped_line)

        flush_stanza()
        return lines

    def _ordered_unique_speakers(
        self,
        dialogue_lines: Sequence[DialogueLine],
    ) -> list[SpeakerVoiceReference]:
        seen: set[str] = set()
        ordered: list[SpeakerVoiceReference] = []
        for line in dialogue_lines:
            key = line.speaker.authored_name.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(line.speaker)
        return ordered


@inject(speaker_map_plan=SpeakerMapPlan)
class ProductionPlan(PlanningNode):
    def __init__(
        self,
        node: ProductionNode,
        script_plans: Sequence[ScriptPlan],
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.script_plans = list(script_plans)

    async def render_node(self) -> ProductionResult:
        await self.speaker_map_plan.render()
        script_results = [await script_plan.render() for script_plan in self.script_plans]
        combined = RenderResult.concatenate(script_results)
        return ProductionResult(
            audio=combined.audio,
            pre_margin=combined.pre_margin,
            post_margin=combined.post_margin,
            pre_gap=combined.pre_gap,
            post_gap=combined.post_gap,
        )
