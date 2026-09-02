from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from carthage.dependency_injection import AsyncInjectable, InjectionKey


CACHE_DIRECTORY_KEY = InjectionKey("cache_dir")
CACHE_OUTPUT_PATH_KEY = InjectionKey("cache_output_path")


def cache_directory_for_output(output_path: str | Path) -> Path:
    """Return the stable ``.wav.cache`` directory for any output encoding."""

    return Path(f"{Path(output_path).with_suffix('.wav')}.cache")


class CachableResource(Protocol):
    """Cache key surface shared by render-request-like cacheable values."""

    def cache_first_words(self) -> str: ...

    def cache_hash(self) -> str: ...


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Stable filename components for one cacheable resource in one collection."""

    collection_name: str
    first_words: str
    resource_hash: str

    @property
    def stem(self) -> str:
        sanitized_label = _sanitize_cache_label(self.first_words)
        return f"{self.collection_name}_{sanitized_label}_{self.resource_hash}"


CacheHit = dict[str, Path]
CacheHitValidator = Callable[[CacheHit], bool]
CacheMissHandler = Callable[[CacheKey, "CacheCollection"], object]


class CacheCollection:
    """Manage one family of sibling cache artifacts under a shared root directory."""

    def __init__(self, name: str, directory: Path | None) -> None:
        self.name = name
        self.directory = directory

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def key_for(self, resource: CachableResource) -> CacheKey:
        return CacheKey(
            collection_name=self.name,
            first_words=resource.cache_first_words(),
            resource_hash=resource.cache_hash(),
        )

    def path_for_subtype(self, key: CacheKey, subtype: str) -> Path:
        if self.directory is None:
            raise RuntimeError(f"Cache collection {self.name!r} is not enabled")
        normalized_subtype = subtype.lstrip(".")
        return self.directory / f"{key.stem}.{normalized_subtype}"

    def find(
        self,
        resource: CachableResource,
        *,
        validate: CacheHitValidator | None = None,
    ) -> CacheHit | None:
        return self._find_by_key(self.key_for(resource), validate=validate)

    def get_or_create(
        self,
        resource: CachableResource,
        on_miss: CacheMissHandler,
        *,
        validate: CacheHitValidator | None = None,
    ) -> CacheHit:
        if self.directory is None:
            raise RuntimeError(f"Cache collection {self.name!r} is not enabled")

        key = self.key_for(resource)
        hit = self._find_by_key(key, validate=validate)
        if hit is not None:
            return hit

        on_miss(key, self)
        hit = self._find_by_key(key, validate=validate)
        if hit is None:
            raise RuntimeError(
                f"Cache miss handler for {key.stem!r} returned without creating cache files"
            )
        return hit

    def delete_hit(self, hit: Mapping[str, Path]) -> None:
        for path in hit.values():
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    def touch_hit(self, hit: Mapping[str, Path]) -> None:
        for path in hit.values():
            os.utime(path, None)

    def _find_by_key(
        self,
        key: CacheKey,
        *,
        validate: CacheHitValidator | None = None,
    ) -> CacheHit | None:
        if self.directory is None:
            return None

        hit = self._scan_hit(key)
        if not hit:
            return None
        if validate is not None and not validate(hit):
            self.delete_hit(hit)
            return None
        self.touch_hit(hit)
        return hit

    def _scan_hit(self, key: CacheKey) -> CacheHit:
        assert self.directory is not None
        prefix = f"{key.stem}."
        hit: CacheHit = {}
        for path in self.directory.glob(f"{key.stem}.*"):
            if not path.is_file():
                continue
            if not path.name.startswith(prefix):
                continue
            subtype = path.suffix.lstrip(".")
            if not subtype:
                continue
            hit[subtype] = path
        return hit


class CacheManager(AsyncInjectable, Mapping[str, CacheCollection]):
    """Lazily materialize named cache collections under the production cache root."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._collections: dict[str, CacheCollection] = {}
        self._root_directory: Path | None = None
        self._root_directory_resolved = False

    def __getitem__(self, cache_type: str) -> CacheCollection:
        collection = self._collections.get(cache_type)
        if collection is None:
            collection = CacheCollection(cache_type, self.root_directory)
            self._collections[cache_type] = collection
        return collection

    def __iter__(self) -> Iterator[str]:
        return iter(self._collections)

    def __len__(self) -> int:
        return len(self._collections)

    @property
    def root_directory(self) -> Path | None:
        if self._root_directory_resolved:
            return self._root_directory

        self._root_directory_resolved = True
        provider_injector = self.ainjector.injector.injector_containing(
            CACHE_DIRECTORY_KEY
        )
        if provider_injector is not None:
            self._root_directory = Path(
                provider_injector.get_instance(CACHE_DIRECTORY_KEY)
            ).expanduser()
            return self._root_directory

        provider_injector = self.ainjector.injector.injector_containing(
            CACHE_OUTPUT_PATH_KEY
        )
        if provider_injector is None:
            self._root_directory = None
            return None
        output_path = Path(
            provider_injector.get_instance(CACHE_OUTPUT_PATH_KEY)
        ).expanduser()
        self._root_directory = cache_directory_for_output(output_path)
        return self._root_directory


def _sanitize_cache_label(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return sanitized or "audio"


__all__ = [
    "CACHE_DIRECTORY_KEY",
    "CACHE_OUTPUT_PATH_KEY",
    "cache_directory_for_output",
    "CacheCollection",
    "CacheHit",
    "CacheKey",
    "CacheManager",
    "CachableResource",
]
