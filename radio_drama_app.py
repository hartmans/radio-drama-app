from __future__ import annotations

import asyncio
import gc
import sys
from pathlib import Path

import soundfile as sf
from carthage.dependency_injection import AsyncInjector

from radio_drama.cli import build_injector_from_namespace, initialize_arg_parser
from radio_drama.debug import reset_debug_outputs
from radio_drama.document import parse_production_file
from radio_drama.errors import DocumentError


def main() -> None:
    parser = initialize_arg_parser(
        "Render a Phase 1 radio-drama XML document to WAV.",
    )
    parser.add_argument("--cut-before", default=None, help="Drop all production audio before the named <mark>.")
    parser.add_argument("--cut-after", default=None, help="Drop all production audio after the named <mark>.")
    args = parser.parse_args()

    async def runner() -> None:
        injector, config, production_path, output_path = build_injector_from_namespace(
            args,
            event_loop=asyncio.get_running_loop(),
        )
        production_node = parse_production_file(production_path)
        reset_debug_outputs(config)
        try:
            ainjector = injector(AsyncInjector)
            production_plan = await production_node.plan(ainjector)
            if args.cut_before is not None:
                production_plan.cut_before_mark(args.cut_before)
                gc.collect()
            if args.cut_after is not None:
                production_plan.cut_after_mark(args.cut_after)
                gc.collect()
            production_result = await production_plan.render()
        finally:
            injector.close()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, production_result.audio, config.resolved_output_sample_rate)

    try:
        asyncio.run(runner())
    except DocumentError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
