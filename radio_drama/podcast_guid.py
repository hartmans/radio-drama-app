"""Generate a stable identifier for a podcast episode's front matter."""

from __future__ import annotations

import uuid


def generate_podcast_guid() -> str:
    """Return a new RFC 4122 UUIDv4 suitable for use as an RSS GUID."""

    return str(uuid.uuid4())


def main() -> None:
    print(generate_podcast_guid())


if __name__ == "__main__":
    main()
