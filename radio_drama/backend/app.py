from __future__ import annotations

import argparse
import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import uvicorn
from carthage.dependency_injection import AsyncInjector
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.io import wavfile

from radio_drama.config import ProductionConfig
from radio_drama.document import parse_production_file
from radio_drama.effects import available_effect_chains, build_named_effect_chain
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
NO_PRESET_NAME = "none"


class PreparePresetsRequest(BaseModel):
    preset_names: list[str] = Field(min_length=1)


class PreparePresetsResponse(BaseModel):
    preset_names: list[str]
    duration_seconds: float
    sample_rate: int


class AudioSliceRequest(BaseModel):
    preset_name: str
    from_time: float = Field(default=0.0, ge=0.0)


class UnknownPresetName(ValueError):
    pass


class PresetNotPrepared(RuntimeError):
    pass


@dataclass(slots=True)
class PresetAudioStore:
    base_result: RenderResult
    sample_rate: int
    prepared_results: dict[str, RenderResult] = field(default_factory=dict)
    _prepare_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.prepared_results.setdefault(NO_PRESET_NAME, self.base_result)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.base_result.frame_count / self.sample_rate

    async def prepare_presets(self, preset_names: Sequence[str]) -> tuple[str, ...]:
        normalized_names = self._normalize_preset_names(preset_names)
        async with self._prepare_lock:
            missing_names = [
                preset_name
                for preset_name in normalized_names
                if preset_name not in self.prepared_results
            ]
            if missing_names:
                rendered_results = await asyncio.gather(
                    *(self._render_preset(preset_name) for preset_name in missing_names)
                )
                for preset_name, result in zip(missing_names, rendered_results, strict=True):
                    self.prepared_results[preset_name] = result
        return normalized_names

    def slice_preset(self, preset_name: str, *, from_time: float) -> RenderResult:
        normalized_name = _normalize_preset_name(preset_name)
        try:
            result = self.prepared_results[normalized_name]
        except KeyError as exc:
            if normalized_name not in set(available_preview_presets()):
                raise UnknownPresetName(normalized_name) from exc
            raise PresetNotPrepared(normalized_name) from exc
        return result.from_time(from_time, sample_rate=self.sample_rate)

    async def _render_preset(self, preset_name: str) -> RenderResult:
        chain = build_named_effect_chain(preset_name)
        return await chain.render(self.base_result, sample_rate=self.sample_rate)

    def _normalize_preset_names(self, preset_names: Sequence[str]) -> tuple[str, ...]:
        normalized_names: list[str] = []
        seen: set[str] = set()
        available = set(available_preview_presets())
        for preset_name in preset_names:
            normalized_name = _normalize_preset_name(preset_name)
            if normalized_name not in available:
                raise UnknownPresetName(normalized_name)
            if normalized_name not in seen:
                seen.add(normalized_name)
                normalized_names.append(normalized_name)
        if not normalized_names:
            raise ValueError("At least one preset name is required")
        return tuple(normalized_names)


def create_app(audio_store: PresetAudioStore) -> FastAPI:
    app = FastAPI(title="Radio Drama Preset Backend")
    app.state.audio_store = audio_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/presets/available")
    async def available_presets() -> dict[str, list[str]]:
        return {"preset_names": list(available_preview_presets())}

    @app.post("/api/presets/prepare", response_model=PreparePresetsResponse)
    async def prepare_presets(request: PreparePresetsRequest) -> PreparePresetsResponse:
        try:
            preset_names = await audio_store.prepare_presets(request.preset_names)
        except UnknownPresetName as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset {exc}. Available presets: {', '.join(available_preview_presets())}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PreparePresetsResponse(
            preset_names=list(preset_names),
            duration_seconds=audio_store.duration_seconds,
            sample_rate=audio_store.sample_rate,
        )

    @app.post("/api/audio-slice")
    async def audio_slice(request: AudioSliceRequest) -> Response:
        try:
            result = audio_store.slice_preset(
                request.preset_name,
                from_time=request.from_time,
            )
        except UnknownPresetName as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset {exc}. Available presets: {', '.join(available_preview_presets())}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PresetNotPrepared as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Preset {exc} has not been prepared. "
                    "POST /api/presets/prepare before requesting audio."
                ),
            ) from exc
        return Response(
            content=render_result_wav_bytes(result, sample_rate=audio_store.sample_rate),
            media_type="audio/wav",
            headers={"X-Render-Start-Time": f"{request.from_time:.6f}"},
        )

    return app


async def render_production_result(
    production_path: str | Path,
    *,
    config: ProductionConfig,
) -> RenderResult:
    production_node = parse_production_file(production_path)
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
    )
    try:
        ainjector = injector(AsyncInjector)
        production_plan = await production_node.plan(ainjector)
        return await production_plan.render()
    finally:
        injector.close()


def render_result_wav_bytes(result: RenderResult, *, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, result.audio)
    return buffer.getvalue()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a preset-preview backend for a radio-drama production XML document.",
    )
    parser.add_argument("production_xml", help="Input production XML file.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port.")
    parser.add_argument("--voice-dir", default=None, help="Directory containing reference voice files.")
    parser.add_argument("--model-file", default=None, help="Path to the VibeVoice model directory.")
    parser.add_argument("--output-sample-rate", type=int, default=None, help="Output sample rate override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum VibeVoice batch size override.")
    parser.add_argument("--device", default=None, help="Preferred torch device override.")
    parser.add_argument("--cfg-scale", type=float, default=None, help="VibeVoice cfg_scale override.")
    parser.add_argument(
        "--disable-prefill",
        action="store_const",
        const=True,
        default=None,
        help="Disable VibeVoice prefill.",
    )
    parser.add_argument(
        "--ddpm-inference-steps",
        type=int,
        default=None,
        help="VibeVoice DDPM inference steps override.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ProductionConfig:
    return ProductionConfig(
        voice_directory=Path(args.voice_dir) if args.voice_dir is not None else None,
        model_name=args.model_file,
        output_sample_rate=args.output_sample_rate,
        output_channels=2,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=args.cfg_scale,
        disable_prefill=args.disable_prefill,
        ddpm_inference_steps=args.ddpm_inference_steps,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    base_result = asyncio.run(
        render_production_result(args.production_xml, config=config)
    )
    audio_store = PresetAudioStore(
        base_result=base_result,
        sample_rate=config.resolved_output_sample_rate,
    )
    uvicorn.run(
        create_app(audio_store),
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _normalize_preset_name(preset_name: str) -> str:
    normalized_name = preset_name.strip().lower()
    if not normalized_name:
        raise ValueError("Preset names cannot be empty")
    return normalized_name


def available_preview_presets() -> tuple[str, ...]:
    return (NO_PRESET_NAME, *available_effect_chains())
