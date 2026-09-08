#!/usr/bin/env python
from __future__ import annotations

import asyncio
import gc
import sys
from collections.abc import Sequence
from pathlib import Path
from carthage.dependency_injection import AsyncInjector

from radio_drama.cli_utils import build_injector_from_namespace, initialize_arg_parser
from radio_drama.debug import reset_debug_outputs
from radio_drama.document import parse_production_file
from radio_drama.errors import DocumentError
from radio_drama.frontmatter import write_audio_file
from radio_drama.production import render_from_input, write_production


def main(argv: Sequence[str] | None = None) -> None:
    parser = initialize_arg_parser(
        "Render a Phase 1 radio-drama XML document to WAV.",
    )
    parser.add_argument(
        "--cut-before",
        default=None,
        help="Drop all production audio before the named <mark>.",
    )
    parser.add_argument(
        "--cut-after",
        default=None,
        help="Drop all production audio after the named <mark>.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Reuse a final WAV instead of rendering the production plan.",
    )
    parser.add_argument(
        "--no-wav",
        action="store_true",
        help="Do not also write WAV when the selected output is another format.",
    )
    args = parser.parse_args(argv)

    async def runner() -> None:
        production_path = Path(args.production_xml)
        production_node = parse_production_file(production_path)
        if args.podcast and production_node.frontmatter.guid is None:
            frontmatter_nodes = production_node.child_elements_named("frontmatter")
            target = frontmatter_nodes[0] if frontmatter_nodes else production_node
            raise target.error("--podcast requires <frontmatter> to include guid")
        injector, config, production_path, output_path = build_injector_from_namespace(
            args,
            event_loop=asyncio.get_running_loop(),
        )
        try:
            reset_debug_outputs(config)
            ainjector = injector(AsyncInjector)
            production_plan = await production_node.plan(ainjector)
            if args.cut_before is not None:
                production_plan.cut_before_mark(args.cut_before)
                gc.collect()
            if args.cut_after is not None:
                production_plan.cut_after_mark(args.cut_after)
                gc.collect()
            if args.input is None:
                production_result = await production_plan.render()
            else:
                production_result = await asyncio.to_thread(
                    render_from_input,
                    args.input,
                    config,
                )
            await write_production(
                production_plan,
                production_result,
                output_path,
                podcast=args.podcast,
            )
            if (
                output_path.suffix.lower() != ".wav"
                and not args.no_wav
                and args.input is None
            ):
                await asyncio.to_thread(
                    write_audio_file,
                    output_path.with_suffix(".wav"),
                    production_result,
                    config.resolved_output_sample_rate,
                    production_node.frontmatter,
                )
        finally:
            injector.close()

    try:
        asyncio.run(runner())
    except DocumentError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
