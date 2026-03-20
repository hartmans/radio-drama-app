from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import yaml
from carthage.dependency_injection import AsyncInjectable, Injector, inject

from .config import ProductionConfig
from .document import DocumentNode, ProductionNode, ScriptNode, SoundNode, SpeakerMapNode, TextNode
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
    """Resolved voice reference for one canonical speaker name."""
    authored_name: str
    voice_name: str
    resolved_path: Path


@dataclass(slots=True)
class DialogueContents:
    """One ordered item inside a script, later addressable by aligned time."""

    start_pos: float = field(default=math.nan, kw_only=True)


@dataclass(slots=True)
class DialogueLine(DialogueContents):
    """Normalized dialogue stanza belonging to one resolved speaker."""
    speaker: SpeakerVoiceReference
    spoken_text: str


@dataclass(frozen=True, slots=True)
class ScriptRenderRequest:
    """Semantic render request sent to a speech resource."""
    normalized_script: str
    voice_samples: tuple[str, ...]


@inject(injector=Injector)
class PlanningNode(AsyncInjectable):
    """Base class for injectable planning objects.

    Planning nodes keep the source ``DocumentNode`` that produced them and
    provide a memoized async ``render()`` entry point so downstream callers do
    not need to coordinate duplicate work themselves.
    """

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


class AudioPlan(PlanningNode):
    """Planning node whose render path produces audio."""

    def __init__(
        self,
        node: DocumentNode | None = None,
        *,
        set_gap: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.pre_margin = 0.0
        self.post_margin = 0.0
        self.pre_gap = 0.0
        self.post_gap = 0.0
        if set_gap and node is not None:
            self.pre_margin = self._timing_attribute_seconds("pre_margin")
            self.post_margin = self._timing_attribute_seconds("post_margin")
            self.pre_gap = self._timing_attribute_seconds("pre_gap")
            self.post_gap = self._timing_attribute_seconds("post_gap")

    async def render(self) -> RenderResult:
        return cast(RenderResult, await super().render())

    async def render_node(self) -> RenderResult:
        raise NotImplementedError

    def leaf_audio_plans(self) -> list["AudioPlan"]:
        return [self]

    def _timing_attribute_seconds(self, attribute_name: str) -> float:
        if self.node is None:
            return 0.0
        raw_value = self.node.attributes.get(attribute_name)
        if raw_value is None:
            return 0.0
        normalized = raw_value.strip()
        if not normalized:
            raise self.document_error(f"{self.node.display_name} {attribute_name} cannot be empty")
        try:
            seconds = float(normalized)
        except ValueError as exc:
            raise self.document_error(
                f"{self.node.display_name} {attribute_name} must be a number of seconds"
            ) from exc
        if seconds < 0:
            raise self.document_error(
                f"{self.node.display_name} {attribute_name} must be non-negative seconds"
            )
        return seconds

    def with_plan_timing(self, result: RenderResult) -> RenderResult:
        return RenderResult(
            audio=result.audio,
            pre_margin=self.pre_margin if self.pre_margin != 0.0 else result.pre_margin,
            post_margin=self.post_margin if self.post_margin != 0.0 else result.post_margin,
            pre_gap=self.pre_gap if self.pre_gap != 0.0 else result.pre_gap,
            post_gap=self.post_gap if self.post_gap != 0.0 else result.post_gap,
        )


@dataclass(slots=True)
class DialogueAudio(DialogueContents):
    """Inline zero-duration audio insertion point within a script."""

    audio_plan: AudioPlan


@inject(config=ProductionConfig)
class SpeakerMapPlan(PlanningNode):
    """Validated speaker map with canonical lookup into resolved voice files."""

    def __init__(self, node: SpeakerMapNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self._voices_by_key: dict[str, SpeakerVoiceReference] = {}

    async def async_ready(self):
        """Parse YAML, validate entries, and resolve voice references."""
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


@inject(config=ProductionConfig, speaker_map_plan=SpeakerMapPlan)
class ScriptPlan(AudioPlan):
    """Plan for one script element and its eventual speech render request."""

    def __init__(self, node: ScriptNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.contents: list[DialogueContents] = []
        self.ordered_speakers: list[SpeakerVoiceReference] = []
        self.render_request: ScriptRenderRequest | None = None
        self._registered_request = None

    async def async_ready(self):
        """Normalize dialogue and register the request with the shared resource."""
        self.contents = await self._parse_contents()
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
        return self.with_plan_timing(await self._registered_request.render())

    @property
    def dialogue_lines(self) -> list[DialogueLine]:
        return [content for content in self.contents if isinstance(content, DialogueLine)]

    @property
    def dialogue_audios(self) -> list[DialogueAudio]:
        return [content for content in self.contents if isinstance(content, DialogueAudio)]

    @classmethod
    async def from_node(cls, ainjector, node: ScriptNode) -> AudioPlan:
        if "preset" in node.attributes and node.preset is None:
            raise node.error("<script> preset attribute cannot be empty")

        script_plan = await ainjector(cls, node=node)
        audio_plan: AudioPlan = script_plan

        if script_plan.dialogue_audios:
            from .forced_alignment import ForcedAlignmentPlan

            audio_plan = await ainjector(
                ForcedAlignmentPlan,
                node=node,
                script_plan=script_plan,
            )

        if node.preset is None:
            return audio_plan

        from .effects import PresetPlan

        return await ainjector(
            PresetPlan,
            node=node,
            audio_plan=audio_plan,
            preset_name=node.preset,
        )

    async def _parse_contents(self) -> list[DialogueContents]:
        contents: list[DialogueContents] = []
        pending_text: list[str] = []

        def flush_pending_text() -> None:
            if not pending_text:
                return
            contents.extend(self._parse_dialogue_text("".join(pending_text)))
            pending_text.clear()

        for child in self.node.children:
            if isinstance(child, TextNode):
                pending_text.append(child.text)
                continue
            flush_pending_text()
            contents.append(DialogueAudio(audio_plan=await child.plan(self.ainjector)))
        flush_pending_text()
        return contents

    def _parse_dialogue_text(self, text: str) -> list[DialogueLine]:
        """Parse speaker stanzas, allowing paragraph continuation lines."""
        text = re.sub(r"^\s*\n", "", text)
        text = re.sub(r"\n\s*$", "", text)
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


@inject(config=ProductionConfig)
class ConcatAudioPlan(AudioPlan):
    """Audio plan that concatenates child render results and consumes child gaps."""

    def __init__(
        self,
        node: DocumentNode,
        audio_plans: Sequence[AudioPlan],
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.audio_plans = list(audio_plans)

    def leaf_audio_plans(self) -> list[AudioPlan]:
        flattened: list[AudioPlan] = []
        for audio_plan in self.audio_plans:
            flattened.extend(audio_plan.leaf_audio_plans())
        return flattened

    async def render_node(self) -> RenderResult:
        rendered_results = [await audio_plan.render() for audio_plan in self.audio_plans]
        return self.with_plan_timing(self._concatenate_results(rendered_results))

    def _concatenate_results(self, results: Sequence[RenderResult]) -> RenderResult:
        if not results:
            return RenderResult.empty(channels=self.config.resolved_output_channels)

        segments: list[np.ndarray] = []
        for index, result in enumerate(results):
            if index == 0 and result.pre_gap > 0:
                segments.append(self._silence_audio(result, result.pre_gap))
            elif index > 0:
                gap_seconds = results[index - 1].post_gap + result.pre_gap
                if gap_seconds > 0:
                    segments.append(self._silence_audio(result, gap_seconds))
            segments.append(result.audio)

        final_result = results[-1]
        if final_result.post_gap > 0:
            segments.append(self._silence_audio(final_result, final_result.post_gap))

        audio = np.concatenate(segments, axis=0) if segments else RenderResult.empty(
            channels=self.config.resolved_output_channels
        ).audio
        return RenderResult(
            audio=audio,
            pre_margin=results[0].pre_margin,
            post_margin=results[-1].post_margin,
        )

    def _silence_audio(self, template: RenderResult, seconds: float) -> np.ndarray:
        frame_count = int(round(seconds * self.config.resolved_output_sample_rate))
        if template.audio.ndim == 1:
            return np.zeros(frame_count, dtype=np.float32)
        return np.zeros((frame_count, template.audio.shape[1]), dtype=np.float32)


@inject(config=ProductionConfig)
class SoundPlan(AudioPlan):
    """Placeholder sound plan that currently renders silence."""

    def __init__(self, node: SoundNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)

    async def render_node(self) -> RenderResult:
        return self.with_plan_timing(RenderResult.empty(channels=self.config.resolved_output_channels))


@inject(config=ProductionConfig, speaker_map_plan=SpeakerMapPlan)
class ProductionPlan(ConcatAudioPlan):
    """Top-level production plan that preserves script order."""

    @property
    def script_plans(self) -> list[AudioPlan]:
        return self.leaf_audio_plans()

    async def render_node(self) -> ProductionResult:
        """Render scripts in document order and concatenate their results."""
        await self.speaker_map_plan.render()
        combined = await super().render_node()
        return ProductionResult(
            audio=combined.audio,
            pre_margin=combined.pre_margin,
            post_margin=combined.post_margin,
            pre_gap=combined.pre_gap,
            post_gap=combined.post_gap,
        )
