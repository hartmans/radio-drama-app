"""Radio-drama proxy adapter for Chatterbox Multilingual V3."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from radio_drama_tts_container import (
    finish_line_work, prepare_line_work, remove_line_work, run_server,
)

LANGUAGE = os.environ.get("CHATTERBOX_LANGUAGE", "en")


class ChatterboxEngine:
    """Keep one V3 model resident and reuse prepared speaker conditionals."""

    def __init__(self) -> None:
        self.model = None
        self._speaker_conditionals: dict[str, Any] = {}

    def load_model(self):
        if self.model is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device="cuda", t3_model="v3"
            )
            torch.set_grad_enabled(False)
        return self.model

    def speaker_conditionals(self, path: str):
        if path not in self._speaker_conditionals:
            model = self.load_model()
            model.prepare_conditionals(path)
            self._speaker_conditionals[path] = model.conds
        return self._speaker_conditionals[path]

    def synthesize_line(self, line: Mapping[str, Any]):
        model = self.load_model()
        model.conds = self.speaker_conditionals(line["speaker"]["voice_path"])
        return model.generate(
            line["spoken_text"],
            language_id=os.environ.get("CHATTERBOX_LANGUAGE", LANGUAGE).lower(),
            temperature=float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.8")),
            cfg_weight=float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.5")),
        )

    def render_batch(self, requests: Sequence[Mapping[str, Any]]):
        import torchaudio

        outputs, work = prepare_line_work(requests)
        try:
            model = self.load_model()
            for item in work:
                torchaudio.save(str(item.path), self.synthesize_line(item.line).cpu(), model.sr)
            return finish_line_work(outputs, work, sample_rate=model.sr)
        finally:
            remove_line_work(work)


def main() -> None:
    run_server(ChatterboxEngine().render_batch)


if __name__ == "__main__":
    main()
