"""Command-line entry point for ``python -m radio_drama.repl``."""

from __future__ import annotations

import argparse

from .console import ReplSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", nargs="?", help="production XML to load before starting")
    args = parser.parse_args(argv)
    session = ReplSession()
    if args.document:
        session.load(args.document)
    session.interact()


if __name__ == "__main__":
    main()
