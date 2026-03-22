from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import ClassVar

import soundfile as sf
from carthage.dependency_injection import AsyncInjectable, inject

from .audio import SUPPORTED_AUDIO_EXTENSIONS, normalize_audio_array
from .config import ProductionConfig
from .document import AttributeOrTextValueNode, AudioPlanContext, ElementContext
from .planning import AudioPlan
from .rendering import RenderResult


@dataclass(frozen=True, slots=True)
class ProductionDocumentPath:
    """Filesystem path for the production XML currently being planned."""

    path: Path


@inject(config=ProductionConfig)
class NormalizedSoundCache(AsyncInjectable):
    """Production-scoped cache of normalized sound assets."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tasks: dict[Path, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def preload(self, sound_path: Path) -> asyncio.Task:
        resolved_path = sound_path.resolve()
        async with self._lock:
            task = self._tasks.get(resolved_path)
            if task is not None and task.done() and task.exception() is not None:
                del self._tasks[resolved_path]
                task = None
            if task is None:
                task = asyncio.create_task(
                    asyncio.to_thread(self._normalize_sound_sync, resolved_path)
                )
                self._tasks[resolved_path] = task
            return task

    def _normalize_sound_sync(self, sound_path: Path):
        with tempfile.TemporaryDirectory(prefix="radio-drama-sound-") as temp_dir:
            output_path = Path(temp_dir) / "normalized.wav"
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(sound_path),
                "-af",
                "loudnorm",
                "-ar",
                str(self.config.resolved_output_sample_rate),
                "-ac",
                str(self.config.resolved_output_channels),
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("ffmpeg is required for sound normalization") from exc
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() or exc.stdout.strip()
                raise RuntimeError(f"ffmpeg sound normalization failed: {stderr}") from exc

            audio, sample_rate = sf.read(
                output_path,
                dtype="float32",
                always_2d=self.config.resolved_output_channels > 1,
            )
        if sample_rate != self.config.resolved_output_sample_rate:
            raise RuntimeError(
                "ffmpeg sound normalization changed the output sample rate unexpectedly"
            )
        if self.config.resolved_output_channels == 1 and getattr(audio, "ndim", 1) == 2:
            audio = audio[:, 0]
        return normalize_audio_array(audio)


@dataclass(slots=True)
class SoundNode(AttributeOrTextValueNode):
    """Document node for an inline named sound reference."""

    tag_name: ClassVar[str] = "sound"
    allow_text: ClassVar[bool] = True
    value_attribute_name: ClassVar[str] = "ref"
    permitted_in_contexts: ClassVar[tuple[ElementContext, ...]] = (AudioPlanContext,)

    @property
    def ref(self) -> str:
        return self.value_from_attribute_or_text

    async def plan(self, ainjector):
        return await ainjector(SoundPlan, node=self)


@inject(config=ProductionConfig, sound_cache=NormalizedSoundCache)
class SoundPlan(AudioPlan):
    """Plan that resolves, normalizes, and renders one sound asset."""

    def __init__(
        self,
        node: SoundNode | None,
        sound_cache: NormalizedSoundCache | None = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.sound_cache = sound_cache
        self.resolved_path: Path | None = None
        self._normalized_audio_task: asyncio.Task | None = None

    async def async_ready(self):
        await self._ensure_normalized_audio_task()
        return await super().async_ready()

    async def render_node(self) -> RenderResult:
        normalized_audio_task = await self._ensure_normalized_audio_task()
        audio = await normalized_audio_task
        return self.with_plan_timing(RenderResult(audio=audio))

    async def _ensure_normalized_audio_task(self) -> asyncio.Task:
        if self._normalized_audio_task is None:
            self.resolved_path = self._resolve_sound_path()
            self._normalized_audio_task = await self.sound_cache.preload(self.resolved_path)
        return self._normalized_audio_task

    def _resolve_sound_path(self) -> Path:
        requested_path = Path(self.node.ref)
        if requested_path.is_absolute():
            if requested_path.is_file() and requested_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                return requested_path
            raise self.document_error(
                f"Sound {self.node.ref!r} was not found as an absolute path"
            )

        sounds_root = self._sounds_root()
        if not sounds_root.is_dir():
            raise self.document_error(f"Sound directory does not exist: {sounds_root}")

        ranked_candidates: list[tuple[int, Path]] = []
        for candidate in _iter_sound_files(sounds_root):
            relative_candidate = candidate.relative_to(sounds_root)
            rank = _sound_match_rank(self.node.ref, relative_candidate)
            if rank is not None:
                ranked_candidates.append((rank, candidate))

        if not ranked_candidates:
            raise self.document_error(
                f"Sound {self.node.ref!r} was not found under {sounds_root}"
            )

        ranked_candidates.sort(key=lambda item: (item[0], item[1].as_posix()))
        best_rank = ranked_candidates[0][0]
        best_candidates = [path for rank, path in ranked_candidates if rank == best_rank]
        if len(best_candidates) > 1:
            relative_paths = ", ".join(
                sorted(str(path.relative_to(sounds_root)) for path in best_candidates)
            )
            raise self.document_error(
                f"Sound {self.node.ref!r} matched multiple files under {sounds_root}: {relative_paths}"
            )
        return best_candidates[0]

    def _sounds_root(self) -> Path:
        if self.config.sounds_directory is not None:
            return self.config.resolved_sounds_directory
        document_path = self._production_document_path()
        return document_path.parent / "sounds"

    def _production_document_path(self) -> Path:
        provider_injector = self.ainjector.injector.injector_containing(ProductionDocumentPath)
        if provider_injector is None:
            raise self.document_error(
                "Relative <sound> references require a production document path in the injector"
            )
        return provider_injector.get_instance(ProductionDocumentPath).path


def _iter_sound_files(sounds_root: Path):
    for current_root, _, filenames in os.walk(sounds_root, followlinks=True):
        current_root_path = Path(current_root)
        for filename in filenames:
            candidate = current_root_path / filename
            if candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                yield candidate


def _sound_match_rank(request_ref: str, candidate_relative_path: Path) -> int | None:
    normalized_ref = PurePosixPath(request_ref.replace("\\", "/"))
    ref_parts_with_suffix = tuple(normalized_ref.parts)
    ref_parts_without_suffix = tuple(normalized_ref.with_suffix("").parts)
    candidate_posix = PurePosixPath(candidate_relative_path.as_posix())
    candidate_parts_with_suffix = tuple(candidate_posix.parts)
    candidate_parts_without_suffix = tuple(candidate_posix.with_suffix("").parts)

    ranks: list[int] = []
    if _path_parts_match_suffix(candidate_parts_with_suffix, ref_parts_with_suffix):
        ranks.append(len(candidate_parts_with_suffix) - len(ref_parts_with_suffix))
    if _path_parts_match_suffix(candidate_parts_without_suffix, ref_parts_without_suffix):
        ranks.append(len(candidate_parts_without_suffix) - len(ref_parts_without_suffix))
    if not ranks:
        return None
    return min(ranks)


def _path_parts_match_suffix(candidate_parts: tuple[str, ...], ref_parts: tuple[str, ...]) -> bool:
    if not ref_parts or len(candidate_parts) < len(ref_parts):
        return False
    return candidate_parts[-len(ref_parts):] == ref_parts
