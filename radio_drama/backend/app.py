from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import uvicorn
from carthage.dependency_injection import AsyncInjector
from fastapi import FastAPI, HTTPException, Response, Depends, staticfiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.io import wavfile

from radio_drama.config import ProductionConfig
from radio_drama.document import parse_production_file
from radio_drama.effects import (
    EffectChainRegistry,
    effect_chain_variables,
)
from radio_drama.expressions import eval_expression
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class StatusResponse(BaseModel):
    """Status endpoint response with preset expressions, base audio info, and duration."""
    preset_expressions: dict[str, str]  # preset_name -> expression string
    base_audio_file: str                # filename only (not full URL)
    total_duration_seconds: float
    sample_rate: int


class ApplyExpressionRequest(BaseModel):
    """Apply expression request with optional from_time for seeking."""
    expression: str
    from_time: float = Field(default=0.0, ge=0.0)


class ApplyExpressionResponse(BaseModel):
    """Apply expression response - returns just the cached filename (frontend constructs URL)."""
    filename: str                    # e.g., "a1b2c3d4e5f6..." (content-aware cache key)
    duration_seconds: float
    sample_rate: int


@dataclass(slots=True)
class ExpressionCacheStore:
    """Maintains cache of rendered audio for expressions.
    
    Cache mapping: sha256(base audio + production presets + expression) -> cached_wav_filename
    All files stored in cache_directory/{sha256}.wav
    
    The frontend accesses these via static mount at /api/cache/{filename}
    and constructs full URLs themselves.
    """
    base_result: RenderResult
    sample_rate: int
    cache_dir: Path
    effect_chains: EffectChainRegistry
    
    _base_audio_filename: str = field(default_factory=lambda: "_base.wav", init=False)
    _base_audio_hash: bytes = field(init=False, repr=False)
    
    def __post_init__(self):
        """Ensure the cache exists and publish this session's base audio."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        base_hasher = hashlib.sha256()
        base_hasher.update(str(self.sample_rate).encode("ascii"))
        base_hasher.update(self.base_result.audio.tobytes())
        for name, expression in self.preset_expressions.items():
            base_hasher.update(b"\0preset\0")
            base_hasher.update(name.encode("utf-8"))
            base_hasher.update(b"\0")
            base_hasher.update(expression.encode("utf-8"))
        self._base_audio_hash = base_hasher.digest()
        self._save_base_audio(self._base_audio_path())
    
    def _base_audio_path(self) -> Path:
        return self.cache_dir / self._base_audio_filename
    
    def _save_base_audio(self, path: Path):
        """Save base audio (no effects applied) to cache."""
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, self.base_result.audio)
        path.write_bytes(buffer.getvalue())
    
    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.base_result.frame_count / self.sample_rate

    @property
    def preset_expressions(self) -> dict[str, str]:
        """Return expressions from the production-scoped preset registry."""

        return {
            name: expression
            for name in self.effect_chains.names()
            if (expression := self.effect_chains.get_expression(name)) is not None
        }
    
    def _expression_sha256(self, expression: str) -> str:
        """Compute a cache key for an expression applied to this base render."""
        hasher = hashlib.sha256(self._base_audio_hash)
        hasher.update(expression.encode("utf-8"))
        return hasher.hexdigest()
    
    async def apply_expression_and_cache(
        self, 
        expression: str, 
        from_time: float = 0.0
    ) -> tuple[str, float, int]:
        """Apply expression to base audio and return cached filename.
        
        If already cached, returns the existing filename without re-rendering.
        Returns (filename, duration_seconds, sample_rate).
        
        Frontend constructs full URL as: /api/cache/{filename}
        And can seek by appending ?from={time} or using HTTP Range requests.
        """
        # Include the base render so changed source or cut bounds cannot reuse stale audio.
        expr_hash = self._expression_sha256(expression)
        cached_path = self.cache_dir / f"{expr_hash}.wav"
        
        if not cached_path.exists():
            # Apply expression to a copy of base audio
            rendered = RenderResult(audio=np.array(self.base_result.audio, copy=True))
            
            # Build effect chain from expression using current preset variables
            variables = effect_chain_variables(self.effect_chains.stages())
            try:
                chain = eval_expression(expression, variables, _effect_chain)
                await asyncio.to_thread(chain.apply, rendered.audio, sample_rate=self.sample_rate)
            except Exception as exc:
                raise ValueError(f"Failed to apply expression: {exc}") from exc
            
            # Save to cache
            buffer = io.BytesIO()
            wavfile.write(buffer, self.sample_rate, rendered.audio)
            cached_path.write_bytes(buffer.getvalue())
        
        return (expr_hash, self.duration_seconds, self.sample_rate)


def _effect_chain(value):
    """Coerce an expression result to the effect-chain interface."""
    from radio_drama.effects import EffectStage
    if isinstance(value, EffectStage):
        return value
    raise TypeError(f"Expected an effect chain, got {type(value).__name__}")


def create_app(
    audio_store: ExpressionCacheStore,
    cache_mount_path: str = "/api/cache",
) -> FastAPI:
    """Create the FastAPI app with expression-based preset endpoints.
    
    Endpoints:
    - GET /api/status: Returns preset_expressions, base_audio_file, total_duration_seconds, sample_rate
    - POST /api/apply-expression: Applies expression, caches result, returns filename
    - GET /api/cache/*: Static files mount for cached audio (frontend constructs URLs)
    
    Security note: The cache directory is publicly readable. Frontend should only expose
    filenames it has explicitly requested. Clients can access any {sha256}.wav if they guess it.
    """
    app = FastAPI(title="Radio Drama Expression Backend")
    
    # Add CORS middleware to allow frontend from port 5173
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://10.36.0.202:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.state.audio_store = audio_store
    
    def get_audio_store() -> ExpressionCacheStore:
        return app.state.audio_store

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status(audio_store: ExpressionCacheStore = Depends(get_audio_store)) -> StatusResponse:
        """Get initial status: preset expressions, base audio filename, and duration.
        
        The frontend should call this once at startup to:
        1. Load all preset names and their expression strings
        2. Get the base audio filename (construct URL as /api/cache/_base.wav)
        3. Know the total duration for UI controls
        """
        return StatusResponse(
            preset_expressions=audio_store.preset_expressions,
            base_audio_file=audio_store._base_audio_filename,
            total_duration_seconds=audio_store.duration_seconds,
            sample_rate=audio_store.sample_rate,
        )

    @app.post("/api/apply-expression", response_model=ApplyExpressionResponse)
    async def apply_expression(
        request: ApplyExpressionRequest,
        audio_store: ExpressionCacheStore = Depends(get_audio_store),
    ) -> ApplyExpressionResponse:
        """Apply an expression to base audio and cache the result.
        
        The expression is evaluated in the context of available effect chain functions
        and production presets. Result is cached as {sha256}.wav in the cache directory.
        
        Returns just the filename - frontend constructs full URL as /api/cache/{filename}
        and can use standard HTTP Range requests for seeking, or construct a from_time
        query parameter: /api/cache/{filename}?from=10.5
        
        Args:
            request.expression: The effect chain expression string
            request.from_time: Starting time in the original audio (currently informational)
        
        Returns:
            filename: SHA256 cache key for the base audio and expression
            duration_seconds: Duration of resulting audio
            sample_rate: Sample rate (same as source)
        """
        try:
            filename, duration, sample_rate = await audio_store.apply_expression_and_cache(
                request.expression, 
                request.from_time
            )
            return ApplyExpressionResponse(
                filename=filename,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to apply expression: {exc}"
            ) from exc

    # Mount cache directory for static file access
    # Frontend accesses cached audio at /api/cache/{filename}
    if audio_store.cache_dir.exists():
        app.mount(
            cache_mount_path,
            staticfiles.StaticFiles(directory=str(audio_store.cache_dir)),
            name="cache",
        )

    return app


async def render_production_result(
    production_path: str | Path,
    *,
    config: ProductionConfig,
    cut_before: str | None = None,
    cut_after: str | None = None,
) -> tuple[RenderResult, EffectChainRegistry]:
    """Render a production and return its production-scoped effect registry."""
    
    production_node = parse_production_file(production_path)
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
        document_path=Path(production_path),
    )
    
    try:
        # Get AsyncInjector and plan/render
        async_injector = injector(AsyncInjector)
        production_plan = await production_node.plan(async_injector)
        effect_chains = production_plan.effect_chains
        if cut_before is not None:
            production_plan.cut_before_mark(cut_before)
            gc.collect()
        if cut_after is not None:
            production_plan.cut_after_mark(cut_after)
            gc.collect()
        base_result = await production_plan.render()
            
        return base_result, effect_chains
            
    finally:
        injector.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an expression-based backend for a radio-drama production XML document.",
    )
    parser.add_argument("production_xml", help="Input production xml file.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port.")
    parser.add_argument("--voice-dir", default=None, help="Directory containing reference voice files.")
    parser.add_argument("--sounds-dir", default=None, help="Directory containing sound files for relative <sound> references.")
    parser.add_argument("--model-file", default=None, help="Path to the VibeVoice model directory.")
    parser.add_argument("--output-sample-rate", type=int, default=None, help="Output sample rate override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum VibeVoice batch size override.")
    parser.add_argument("--cut-before", default=None, help="Drop all production audio before the named <mark>.")
    parser.add_argument("--cut-after", default=None, help="Drop all production audio after the named <mark>.")
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
    parser.add_argument("--cache-dir", default=None, help="Cache directory for rendered expressions.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ProductionConfig:
    return ProductionConfig(
        voice_directory=Path(args.voice_dir) if args.voice_dir is not None else None,
        sounds_directory=Path(args.sounds_dir) if args.sounds_dir is not None else None,
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
    
    # Render production to get base audio
    base_result, effect_chains = asyncio.run(
        render_production_result(
            args.production_xml,
            config=config,
            cut_before=args.cut_before,
            cut_after=args.cut_after,
        )
    )
    
    # Determine cache directory
    if args.cache_dir:
        cache_path = Path(args.cache_dir).expanduser()
    else:
        # Default cache in same directory as production file
        cache_path = Path(args.production_xml).parent / "cache"
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    audio_store = ExpressionCacheStore(
        base_result=base_result,
        sample_rate=config.resolved_output_sample_rate,
        cache_dir=cache_path,
        effect_chains=effect_chains,
    )
    
    uvicorn.run(
        create_app(audio_store),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
