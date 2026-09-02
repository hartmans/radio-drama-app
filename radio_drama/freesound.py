"""Generate Markdown credits for Freesound assets used by a production."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml
from carthage.dependency_injection import AsyncInjector

from .cli import build_injector_from_namespace, initialize_arg_parser
from .document import parse_production_file
from .errors import DocumentError
from .sound import SoundPlan, sound_plans_in


FREESOUND_API_KEY_PATH = Path("~/freesound_api_key")
_FREESOUND_FILENAME = re.compile(r"^(?P<id>\d+)_")


@dataclass(frozen=True, slots=True)
class FreesoundCredit:
    """Attribution fields returned for one Freesound asset."""

    sound_id: int
    name: str
    username: str


def freesound_id(sound: SoundPlan) -> int | None:
    """Extract a Freesound ID from a resolved or authored sound filename."""

    source = sound.resolved_path or Path(sound.node.ref)
    match = _FREESOUND_FILENAME.match(source.name)
    return int(match.group("id")) if match else None


def likely_freesound_ids(plan) -> tuple[int, ...]:
    """Return unique Freesound IDs used by a plan, in first-use order."""

    sound_ids: list[int] = []
    seen: set[int] = set()
    for sound in sound_plans_in(plan):
        sound_id = freesound_id(sound)
        if sound_id is not None and sound_id not in seen:
            seen.add(sound_id)
            sound_ids.append(sound_id)
    return tuple(sound_ids)


def load_api_key(path: Path = FREESOUND_API_KEY_PATH) -> str:
    """Load a token from ``FREESOUND_API_KEY`` or the conventional YAML file.

    Freesound labels the token-auth value as both the client secret and API
    key.  The YAML file may retain the accompanying ``client_id`` for future
    OAuth use, but read-only token authentication only uses ``secret``.
    """

    key = os.environ.get("FREESOUND_API_KEY")
    if key is None:
        try:
            credentials = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Freesound API key not found; set FREESOUND_API_KEY or create "
                f"{path}"
            ) from exc
        if not isinstance(credentials, dict) or not isinstance(
            credentials.get("secret"), str
        ):
            raise RuntimeError(
                f"Freesound credentials in {path} must be YAML containing a secret"
            )
        key = credentials["secret"]
    key = key.strip()
    if not key:
        raise RuntimeError("Freesound API key is empty")
    return key


def lookup_sound(sound_id: int, api_key: str) -> FreesoundCredit:
    """Fetch the public attribution fields for one sound from Freesound."""

    url = f"https://freesound.org/apiv2/sounds/{sound_id}/?fields=id,name,username"
    request = Request(url, headers={"Authorization": f"Token {api_key}"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Freesound lookup failed for sound {sound_id}: {exc}") from exc
    return FreesoundCredit(
        sound_id=int(payload["id"]),
        name=str(payload["name"]),
        username=str(payload["username"]),
    )


def markdown_credits(credits: Iterable[FreesoundCredit]) -> str:
    """Format Freesound attribution as a Markdown section."""

    lines = ["## Sound credits", ""]
    for credit in credits:
        sound_url = f"https://freesound.org/s/{credit.sound_id}/"
        author_url = f"https://freesound.org/people/{quote(credit.username, safe='')}/"
        lines.append(
            f"- [{credit.name}]({sound_url}) by "
            f"[{credit.username}]({author_url}) on Freesound"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None):
    parser = initialize_arg_parser(
        "Plan a production and print Markdown credits for its Freesound assets."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    async def runner() -> str:
        injector, _, production_path, _ = build_injector_from_namespace(
            args, event_loop=asyncio.get_running_loop()
        )
        try:
            plan = await parse_production_file(production_path).plan(injector(AsyncInjector))
            sound_ids = likely_freesound_ids(plan)
            if not sound_ids:
                return markdown_credits(())
            api_key = load_api_key()
            credits = await asyncio.gather(
                *(
                    asyncio.to_thread(lookup_sound, sound_id, api_key)
                    for sound_id in sound_ids
                )
            )
            return markdown_credits(credits)
        finally:
            injector.close()

    try:
        sys.stdout.write(asyncio.run(runner()))
    except (DocumentError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
