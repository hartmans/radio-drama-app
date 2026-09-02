"""Backend-independent persistent cache and timing for TTS requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import soundfile as sf

from .audio import convert_audio_format
from .rendering import BackendTtsResult, DialogueLineTiming, ScriptRenderResult, ScriptTiming


if TYPE_CHECKING:
    from .dialogue import ScriptEvent, ScriptRenderRequest, TtsResource


_ALIGNMENT_VERSION = "script-timing-v1"


@dataclass(slots=True)
class CachedTtsRequest:
    """One lazy backend request mediated by the shared TTS cache."""

    resource: "TtsResource"
    request: "ScriptRenderRequest"
    _result_task: asyncio.Task[ScriptRenderResult] | None = None
    _backend_result: BackendTtsResult | None = None
    _wav_path: Path | None = None
    _meta_path: Path | None = None
    _alignment_key: str | None = None

    async def render(self) -> ScriptRenderResult:
        if self._result_task is None:
            self._result_task = asyncio.create_task(self._render())
        try:
            return await self._result_task
        except BaseException:
            self._result_task = None
            raise

    async def ensure_timing(
        self,
        contents: Sequence["ScriptEvent"],
        result: ScriptRenderResult,
    ) -> ScriptTiming:
        await self.render()
        assert self._backend_result is not None
        timing = self._backend_result.timing
        if timing is not None and self._alignment_key is not None:
            if self._alignment_key.startswith("native:"):
                return timing
            requested_key = self._forced_alignment_key(contents)
            if self._alignment_key == requested_key:
                return timing

        from .forced_alignment import WhisperXResource

        whisperx = await self.resource.ainjector.get_instance_async(WhisperXResource)
        timing = await whisperx.script_timing(contents, result)
        self._backend_result.timing = timing
        self._alignment_key = self._forced_alignment_key(contents)
        await asyncio.to_thread(self._write_metadata)
        return timing

    async def _render(self) -> ScriptRenderResult:
        if not self.request.dialogue_lines:
            return ScriptRenderResult.empty(
                channels=self.resource.config.resolved_output_channels
            )
        cached = await asyncio.to_thread(self._load_cached)
        if cached is None:
            registration = await self.resource.register_backend_request(self.request)
            backend_result = await registration.render()
            self._backend_result = backend_result
            await asyncio.to_thread(self._store_backend_result)
            persisted = await asyncio.to_thread(self._load_cached)
            if persisted is not None:
                self._backend_result = persisted
        else:
            self._backend_result = cached

        backend_result = self._backend_result
        assert backend_result is not None
        return ScriptRenderResult(
            audio=convert_audio_format(
                backend_result.audio,
                input_sample_rate=backend_result.sample_rate,
                output_sample_rate=self.resource.config.resolved_output_sample_rate,
                output_channels=self.resource.config.resolved_output_channels,
            ),
            timing=backend_result.timing,
        )

    def _cache_paths(self) -> tuple[Path, Path] | None:
        collection = self.resource.cache_manager[self.resource.cache_collection_name]
        if not collection.enabled:
            return None
        key = collection.key_for(self.request)
        return (
            collection.path_for_subtype(key, "wav"),
            collection.path_for_subtype(key, "meta"),
        )

    def _load_cached(self) -> BackendTtsResult | None:
        paths = self._cache_paths()
        if paths is None:
            return None
        wav_path, meta_path = paths
        self._wav_path = wav_path
        self._meta_path = meta_path
        if not wav_path.is_file() or not meta_path.is_file():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            sample_rate = int(payload["sample_rate"])
            timing = _timing_from_payload(payload, len(self.request.dialogue_lines))
            audio, actual_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        if int(actual_rate) != sample_rate:
            return None
        self._alignment_key = payload.get("alignment_key") if timing is not None else None
        return BackendTtsResult(audio=audio, sample_rate=sample_rate, timing=timing)

    def _store_backend_result(self) -> None:
        assert self._backend_result is not None
        if (
            self._backend_result.timing is not None
            and len(self._backend_result.timing.dialogue_lines)
            != len(self.request.dialogue_lines)
        ):
            raise RuntimeError(
                "TTS backend timing must contain one span per dialogue line"
            )
        paths = self._cache_paths()
        if paths is None:
            return
        wav_path, meta_path = paths
        self._wav_path = wav_path
        self._meta_path = meta_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        reusable = self._backend_result.cache_wav_path
        if reusable is None or reusable.resolve() != wav_path.resolve():
            sf.write(wav_path, self._backend_result.audio, self._backend_result.sample_rate)
        if self._backend_result.timing is not None:
            self._alignment_key = f"native:{self._audio_identity()}"
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self._meta_path is None or self._backend_result is None:
            return
        payload = {
            "sample_rate": self._backend_result.sample_rate,
            "frame_count": self._backend_result.frame_count,
            "channels": self._backend_result.channel_count,
            "alignment_key": self._alignment_key,
            "dialogue_line_spans": (
                [
                    [line.start, line.end]
                    for line in self._backend_result.timing.dialogue_lines
                ]
                if self._backend_result.timing is not None
                else None
            ),
        }
        temporary = self._meta_path.with_suffix(".meta.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._meta_path)

    def _audio_identity(self) -> str:
        if self._wav_path is None or not self._wav_path.is_file():
            assert self._backend_result is not None
            return f"memory:{self._backend_result.frame_count}:{self._backend_result.sample_rate}"
        stat = self._wav_path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _forced_alignment_key(self, contents: Sequence["ScriptEvent"]) -> str:
        from .dialogue import DialogueLine, ScriptGap

        projection = []
        for content in contents:
            if isinstance(content, DialogueLine):
                projection.append(
                    {
                        "type": "line",
                        "text": content.spoken_text,
                        "source": content.source,
                        "handling": content.handling,
                    }
                )
            elif isinstance(content, ScriptGap):
                projection.append(
                    {"type": "gap", "label": content.label, "mode": content.mode}
                )
        encoded = json.dumps(projection, sort_keys=True, ensure_ascii=True)
        projection_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{_ALIGNMENT_VERSION}:{self._audio_identity()}:{projection_hash}"


def _timing_from_payload(payload: dict, expected_lines: int) -> ScriptTiming | None:
    spans = payload.get("dialogue_line_spans")
    if spans is None or payload.get("alignment_key") is None:
        return None
    if not isinstance(spans, list) or len(spans) != expected_lines:
        return None
    try:
        lines = tuple(
            DialogueLineTiming(start=float(span[0]), end=float(span[1]))
            for span in spans
            if isinstance(span, list) and len(span) == 2
        )
    except (TypeError, ValueError):
        return None
    if len(lines) != expected_lines:
        return None
    return ScriptTiming(dialogue_lines=lines)


__all__ = ["CachedTtsRequest"]
