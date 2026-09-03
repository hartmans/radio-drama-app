#!/usr/bin/env python
from __future__ import annotations

import asyncio
import gc
import sys
from collections.abc import Sequence
from pathlib import Path
from carthage.dependency_injection import AsyncInjector

from radio_drama.cli import build_injector_from_namespace, initialize_arg_parser
from radio_drama.debug import reset_debug_outputs
from radio_drama.document import parse_production_file
from radio_drama.errors import DocumentError
from radio_drama.production import write_production


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
            production_result = await production_plan.render()
            await write_production(
                production_plan,
                production_result,
                output_path,
                podcast=args.podcast,
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
