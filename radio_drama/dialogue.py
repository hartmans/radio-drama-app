from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import yaml
from carthage.dependency_injection import ExistingProvider, Injector, inject

from .audio import AudioPlan, ComposeAudioPlan, SUPPORTED_AUDIO_EXTENSIONS
from .config import ProductionConfig
from .planning import AudioAttrValue, PRODUCTION_PLANNING_INJECTOR_KEY, PlanningNode
from .rendering import RenderResult


if TYPE_CHECKING:
    from .document import (
        DocumentNode,
        GroupNode,
        IgnoreNode,
        LineNode,
        ScriptGapNode,
        ScriptNode,
        SpeakerMapNode,
        TextNode,
    )


_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s*(.*)$")


@dataclass(frozen=True, slots=True)
class SpeakerVoiceReference:
    """Resolved voice reference for one canonical speaker name."""

    authored_name: str
    voice_name: str
    resolved_path: Path


@dataclass(slots=True)
class ScriptEvent:
    """One ordered authored event inside a script timeline.

    ``start_pos`` is measured in seconds in the rendered script timeline.
    It may remain ``NaN`` until alignment or backend-native script timing data
    resolves it.
    """

    start_pos: float = field(default=math.nan, kw_only=True)


@dataclass(slots=True)
class DialogueContent(ScriptEvent):
    """Script event that affects speech synthesis or forced alignment."""


@dataclass(slots=True)
class DialogueLine(DialogueContent):
    """Normalized dialogue stanza belonging to one resolved speaker."""

    speaker: SpeakerVoiceReference
    spoken_text: str
    handling: Literal["normal", "ignore", "special"] = "normal"
    source: Literal["tts", "recording"] = "tts"
    node: DocumentNode | None = None


@dataclass(slots=True)
class ScriptGap(DialogueContent):
    """Explicit omitted region where forced alignment should resynchronize."""

    label: str = ""


@dataclass(frozen=True, slots=True, init=False)
class ScriptRenderRequest:
    """Semantic render request sent to a speech resource.

    The request is also the stable cache identity for script-level speech
    output. It owns the semantic serialization that both speech backends use
    when deriving cache filenames and adjacent JSON metadata.
    """

    dialogue_contents: list[DialogueContent]
    first_words: str = ""

    def __init__(
        self,
        dialogue_contents: Sequence[DialogueContent] | None = None,
        *,
        dialogue_lines: Sequence[DialogueLine] | None = None,
        first_words: str = "",
    ) -> None:
        if dialogue_contents is None:
            dialogue_contents = [] if dialogue_lines is None else list(dialogue_lines)
        elif dialogue_lines is not None:
            raise TypeError("Specify either dialogue_contents or dialogue_lines, not both")
        object.__setattr__(self, "dialogue_contents", list(dialogue_contents))
        object.__setattr__(self, "first_words", first_words)

    @property
    def dialogue_lines(self) -> list[DialogueLine]:
        return [
            content for content in self.dialogue_contents if isinstance(content, DialogueLine)
        ]

    def cache_first_words(self) -> str:
        """Return the human-authored cache label prior to filename sanitization."""

        label = " ".join(self.first_words.split()).strip()
        if not label:
            label = self._fallback_cache_label()
        return label[:40] or "empty-script"

    def cache_hash(self) -> str:
        """Return the stable semantic hash used for cache filenames."""

        payload = json.dumps(
            self.serialize_cache_request(),
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_cache_hit(self, hit: Mapping[str, Path]) -> bool:
        """Require the common render cache artifacts for one script request."""

        return {"json", "wav"}.issubset(hit)

    def serialize_cache_request(self) -> dict[str, object]:
        """Return the stable request payload shared by both speech backends."""

        return {
            "dialogue_lines": [
                {
                    "speaker": line.speaker.authored_name,
                    "voice_name": line.speaker.voice_name,
                    "voice_path": str(line.speaker.resolved_path),
                    "spoken_text": line.spoken_text,
                    "handling": line.handling,
                    "source": line.source,
                }
                for line in self.dialogue_lines
            ],
            "first_words": self.first_words,
        }

    def build_cache_payload(self, **metadata: object) -> dict[str, object]:
        """Return the adjacent JSON payload for one cached render result."""

        payload = self.serialize_cache_request()
        payload.update(metadata)
        return payload

    def _fallback_cache_label(self) -> str:
        for line in self.dialogue_lines:
            stripped_line = " ".join(line.spoken_text.split()).strip()
            if stripped_line:
                return stripped_line
        return ""


@dataclass(slots=True)
class DialogueAudio(ScriptEvent):
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
        production_injector = self._production_injector()
        if production_injector is not None:
            try:
                production_injector.add_provider(self)
            except ExistingProvider as exc:
                raise self.document_error("A <production> may contain only one <speaker-map>") from exc
        return await super().async_ready()

    def _production_injector(self) -> Injector | None:
        provider_injector = self.ainjector.injector.injector_containing(PRODUCTION_PLANNING_INJECTOR_KEY)
        if provider_injector is None:
            return None
        return provider_injector.get_instance(PRODUCTION_PLANNING_INJECTOR_KEY)

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
            if not child.is_file() or child.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
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


@inject(config=ProductionConfig)
class ScriptPlan(AudioPlan):
    """Plan for one script element and its eventual speech render request."""

    def __init__(self, node: ScriptNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.speaker_map_plan: SpeakerMapPlan | None = None
        self.script_events: list[ScriptEvent] = []
        self.ordered_speakers: list[SpeakerVoiceReference] = []
        self.render_request: ScriptRenderRequest | None = None
        self._registered_request = None

    def __repr__(self) -> str:
        line = self.node.location.line
        dialogue_lines = [
            content for content in self.dialogue_contents if isinstance(content, DialogueLine)
        ]
        first_words = self._preferred_first_words(dialogue_lines)
        if line is None:
            if first_words:
                return f"ScriptPlan(line=unknown, first_words={first_words!r})"
            return "ScriptPlan(line=unknown)"
        if first_words:
            return f"ScriptPlan(line={line}, first_words={first_words!r})"
        return f"ScriptPlan(line={line})"

    async def async_ready(self):
        """Normalize dialogue and prepare the base audio render path."""

        self.speaker_map_plan = self._require_speaker_map_plan()
        self.script_events = await self._parse_script_events()
        if (
            any(
                isinstance(event, DialogueLine) and event.source == "recording"
                for event in self.script_events
            )
            and self.node.recording_node is None
        ):
            raise self.document_error(
                "Recorded dialogue requires a leading <recording> declaration"
            )
        dialogue_lines = [
            content for content in self.dialogue_contents if isinstance(content, DialogueLine)
        ]
        self.ordered_speakers = self._ordered_unique_speakers(dialogue_lines)
        if self._source_has_output("tts"):
            first_words = self._preferred_first_words(dialogue_lines)
            self.render_request = ScriptRenderRequest(
                dialogue_contents=list(self.dialogue_contents),
                first_words=first_words,
            )
            await self.register_render_request()
        return await super().async_ready()

    def _tts_resource_type(self):
        if self.node.tts == "qwen":
            from .qwen_tts import QwenTtsResource

            return QwenTtsResource
        from .vibevoice import VibeVoiceResource

        return VibeVoiceResource

    def _require_speaker_map_plan(self) -> SpeakerMapPlan:
        provider_injector = self.ainjector.injector.injector_containing(SpeakerMapPlan)
        if provider_injector is None:
            raise self.document_error(
                "A <script> requires a <speaker-map> to be planned before it"
            )
        return provider_injector.get_instance(SpeakerMapPlan)

    async def register_render_request(self) -> None:
        resource = await self.ainjector.get_instance_async(self._tts_resource_type())
        self._registered_request = await resource.register_request(self.render_request)

    async def render_base_audio(self) -> RenderResult:
        if self._registered_request is None:
            return RenderResult.empty(channels=self.config.resolved_output_channels)
        return await self._registered_request.render()

    async def render_node(self) -> RenderResult:
        return await self.render_base_audio()

    async def layout_node(self) -> None:
        result = await self.render_base_audio()
        self.inner_last = self._frames_to_seconds(result.frame_count)
        self.advance = self.inner_last

    @property
    def dialogue_contents(self) -> list[DialogueContent]:
        return [content for content in self.script_events if isinstance(content, DialogueContent)]

    @property
    def dialogue_audios(self) -> list[DialogueAudio]:
        return [content for content in self.script_events if isinstance(content, DialogueAudio)]

    @classmethod
    async def from_node(cls, ainjector, node: ScriptNode, **kwargs) -> AudioPlan:
        node.tts
        script_plan = await ainjector(
            cls,
            node=node,
            attrs={} if node.element_children else None,
            **kwargs,
        )
        audio_plan: AudioPlan = script_plan

        if script_plan.needs_source_slicing():
            audio_plan = await cls._build_aligned_audio_plan(
                ainjector,
                node,
                script_plan,
                attrs=type(script_plan).attrs_from_node(node),
            )
        elif node.element_children:
            # A declaration is not rendered inline, but it still means the
            # script plan was created without outer attributes so a later
            # retained source could own them. Keep those attrs on one outer
            # audio plan when the declaration is ultimately unused.
            audio_plan = await ainjector(
                ComposeAudioPlan,
                node=node,
                audio_plans=[script_plan],
                attrs=type(script_plan).attrs_from_node(node),
            )
        return audio_plan

    @classmethod
    async def _build_aligned_audio_plan(
        cls,
        ainjector,
        node: ScriptNode,
        script_plan: ScriptPlan,
        *,
        attrs: Mapping[str, AudioAttrValue],
    ) -> AudioPlan:
        from .forced_alignment import AlignedScriptSource, ScriptSlice

        sources = await cls._aligned_sources(ainjector, node, script_plan)
        audio_plans: list[AudioPlan] = []
        content_index = 0

        while content_index < len(script_plan.script_events):
            content = script_plan.script_events[content_index]
            if isinstance(content, DialogueAudio):
                audio_plans.append(content.audio_plan)
                content_index += 1
                continue
            if isinstance(content, ScriptGap):
                content_index += 1
                continue
            source = content.source
            aligned_script_source, marker_indexes = sources[source]
            if content.handling == "special":
                audio_plans.append(
                    await ainjector(
                        ScriptSlice,
                        node=content.node,
                        aligned_script_source=aligned_script_source,
                        start_marker=marker_indexes[content_index],
                        end_marker=marker_indexes[content_index + 1],
                        name=content.spoken_text[:30] or content.speaker.authored_name,
                    )
                )
                content_index += 1
                continue
            if content.handling != "normal":
                content_index += 1
                continue
            end_index = script_plan.advance_normal_dialogue_slice_end(content_index, source)
            audio_plans.append(
                await ainjector(
                    ScriptSlice,
                    node=node,
                    attrs={},
                    aligned_script_source=aligned_script_source,
                    start_marker=marker_indexes[content_index],
                    end_marker=marker_indexes[end_index],
                    name=cls._script_slice_name(script_plan.script_events, content_index),
                )
            )
            content_index = end_index

        if not audio_plans:
            return await ainjector(ComposeAudioPlan, node=node, audio_plans=[], attrs=attrs)
        return await ainjector(
            ComposeAudioPlan,
            node=node,
            audio_plans=audio_plans,
            attrs=attrs,
        )

    @classmethod
    async def _aligned_sources(cls, ainjector, node, script_plan):
        """Create only the source-local alignment graphs selected by output lines."""
        from .forced_alignment import AlignedScriptSource
        from .sound import SoundPlan

        sources = {}
        if script_plan._source_has_output("tts"):
            projection = list(script_plan.script_events)
            sources["tts"] = (
                await ainjector(
                    AlignedScriptSource,
                    node=node,
                    audio_provider=script_plan,
                    contents=projection,
                ),
                cls._projection_marker_indexes(script_plan.script_events, projection),
            )
        if script_plan._source_has_output("recording"):
            recording = node.recording_node
            if recording is None:
                raise node.error("Recorded dialogue requires a leading <recording> declaration")
            recording_plan = await ainjector(
                SoundPlan, node=recording, attrs=SoundPlan.attrs_from_node(recording)
            )
            projection = script_plan.recording_projection()
            sources["recording"] = (
                await ainjector(
                    AlignedScriptSource,
                    node=recording,
                    audio_provider=recording_plan,
                    contents=projection,
                ),
                cls._projection_marker_indexes(script_plan.script_events, projection),
            )
        return sources

    @staticmethod
    def _projection_marker_indexes(authored, projection):
        """Map authored boundaries to local projection marker indexes.

        A missing authored event intentionally maps to the next source-local
        boundary; source transitions therefore retain next-start slice ends.
        """
        local_by_id = {id(event): index for index, event in enumerate(projection)}
        indexes = []
        for boundary in range(len(authored) + 1):
            local = next(
                (local_by_id[id(event)] for event in authored[boundary:] if id(event) in local_by_id),
                len(projection),
            )
            indexes.append(local)
        return indexes

    @staticmethod
    def _script_slice_name(
        script_events: Sequence[ScriptEvent],
        marker_index: int,
    ) -> str:
        if marker_index >= len(script_events):
            return "script end"
        for content in script_events[marker_index:]:
            if isinstance(content, DialogueLine) and content.handling == "normal":
                return content.spoken_text[:30]
            if isinstance(content, ScriptGap):
                return content.label[:30] or "script gap"
            if isinstance(content, DialogueAudio):
                return repr(content.audio_plan)
        return "ignored script"

    async def _parse_script_events(self) -> list[ScriptEvent]:
        from .document import GroupNode, IgnoreNode, LineNode, RecordingNode, ScriptGapNode, TextNode
        from .forced_alignment import ScriptSlice

        contents: list[ScriptEvent] = []
        pending_text: list[str] = []
        pending_text_node: TextNode | None = None

        def flush_pending_text() -> None:
            nonlocal pending_text_node
            if not pending_text:
                return
            contents.extend(
                self._parse_dialogue_text(
                    "".join(pending_text),
                    error_node=pending_text_node,
                )
            )
            pending_text.clear()
            pending_text_node = None

        for child in self.node.children:
            if isinstance(child, TextNode):
                if pending_text_node is None:
                    pending_text_node = child
                pending_text.append(child.text)
                continue
            flush_pending_text()
            if isinstance(child, RecordingNode):
                continue
            if isinstance(child, IgnoreNode):
                contents.extend(
                    self._parse_dialogue_text(
                        child.text_content,
                        handling="ignore",
                        error_node=child,
                    )
                )
                continue
            if isinstance(child, GroupNode):
                contents.extend(
                    self._parse_dialogue_text(
                        child.text_content,
                        handling="special",
                        node=child,
                        error_node=child,
                    )
                )
                continue
            if isinstance(child, LineNode):
                line_attrs = ScriptSlice.attrs_from_node(child)
                handling: Literal["normal", "special"] = "special" if line_attrs else "normal"
                contents.append(
                    DialogueLine(
                        speaker=self.speaker_map_plan.lookup(child.speaker),
                        spoken_text=child.normalized_text_content,
                        handling=handling,
                        source=child.source,
                        node=child if handling == "special" else None,
                    )
                )
                continue
            if isinstance(child, ScriptGapNode):
                contents.append(ScriptGap(label=child.label))
                continue
            contents.append(DialogueAudio(audio_plan=await child.plan(self.ainjector)))
        flush_pending_text()
        return contents

    def _parse_dialogue_text(
        self,
        text: str,
        *,
        handling: Literal["normal", "ignore", "special"] = "normal",
        node: DocumentNode | None = None,
        error_node: DocumentNode | None = None,
    ) -> list[DialogueLine]:
        """Parse dialogue text into speaker-scoped lines.

        A new ``DialogueLine`` starts only when the parser encounters a
        non-empty line of the form ``speaker: ...`` and that speaker resolves
        in the current ``SpeakerMapPlan``. Blank lines end only the current
        paragraph, not the current ``DialogueLine``; continuation lines are
        folded into the current speaker's text. The current ``DialogueLine``
        is therefore finalized only by a new recognized speaker stanza or by
        reaching the end of this text chunk.
        """

        text = re.sub(r"^\s*\n", "", text)
        text = re.sub(r"\n\s*$", "", text)
        if not text:
            return []

        lines: list[DialogueLine] = []
        current_speaker: SpeakerVoiceReference | None = None
        current_source: Literal["tts", "recording"] = "tts"
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
                lines.append(
                    DialogueLine(
                        speaker=current_speaker,
                        spoken_text=spoken_text,
                        handling=handling,
                        source=current_source,
                        node=node,
                    )
                )

        for raw_line in text.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line:
                flush_paragraph()
                continue
            match = _SPEAKER_LINE_RE.match(stripped_line)
            if match is not None:
                candidate_speaker = match.group(1).strip()
                candidate_source: Literal["tts", "recording"] = "tts"
                if candidate_speaker.startswith("~"):
                    candidate_source = "recording"
                    candidate_speaker = candidate_speaker[1:].strip()
                try:
                    speaker_ref = self.speaker_map_plan.lookup(candidate_speaker)
                except KeyError:
                    speaker_ref = None
                if speaker_ref is not None:
                    flush_stanza()
                    current_speaker = speaker_ref
                    current_source = candidate_source
                    current_paragraph.append(match.group(2).strip())
                    continue
            if current_speaker is None:
                raise self.document_error(
                    "Scripts may begin only with a recognized `speaker:` stanza",
                    node=error_node or node,
                )
            current_paragraph.append(stripped_line)

        flush_stanza()
        return lines

    def needs_source_slicing(self) -> bool:
        return any(
            isinstance(content, DialogueAudio)
            or isinstance(content, ScriptGap)
            or (
                isinstance(content, DialogueLine)
                and (content.handling != "normal" or content.source != "tts")
            )
            for content in self.script_events
        )

    def next_non_audio_index(self, start_index: int) -> int:
        index = start_index
        while index < len(self.script_events) and isinstance(self.script_events[index], DialogueAudio):
            index += 1
        return index

    def advance_normal_dialogue_slice_end(self, start_index: int, source: str) -> int:
        index = start_index
        while index < len(self.script_events):
            content = self.script_events[index]
            if isinstance(content, DialogueAudio):
                break
            if isinstance(content, ScriptGap):
                # A gap is an authored end boundary for the preceding run;
                # the following line starts a new aligned run after resync.
                break
            if (
                not isinstance(content, DialogueLine)
                or content.handling != "normal"
                or content.source != source
            ):
                break
            index += 1
        return index

    def _source_has_output(self, source: str) -> bool:
        return any(
            isinstance(event, DialogueLine)
            and event.source == source
            and event.handling != "ignore"
            for event in self.script_events
        )

    def recording_projection(self) -> list[ScriptEvent]:
        """Return recording-local alignment events without TTS-only transcript."""
        retained: list[ScriptEvent] = []
        has_recording = any(
            isinstance(event, DialogueLine) and event.source == "recording"
            for event in self.script_events
        )
        if not has_recording:
            return retained
        for event in self.script_events:
            if isinstance(event, DialogueLine) and event.source == "recording":
                retained.append(event)
            elif isinstance(event, ScriptGap):
                retained.append(event)
            elif isinstance(event, DialogueAudio):
                retained.append(event)
        return retained

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

    @staticmethod
    def _preferred_first_words(
        dialogue_lines: Sequence[DialogueLine],
    ) -> str:
        preferred_line = next(
            (
                line
                for line in dialogue_lines
                if line.handling == "normal" and line.spoken_text.strip()
            ),
            None,
        )
        if preferred_line is None:
            preferred_line = next(
                (line for line in dialogue_lines if line.spoken_text.strip()),
                None,
            )
        if preferred_line is None:
            return "empty-script"
        label = " ".join(preferred_line.spoken_text.split()).strip()
        return label[:40] or "empty-script"


__all__ = [
    "DialogueContent",
    "DialogueAudio",
    "DialogueLine",
    "ScriptEvent",
    "ScriptGap",
    "ScriptPlan",
    "ScriptRenderRequest",
    "SpeakerMapPlan",
    "SpeakerVoiceReference",
]
