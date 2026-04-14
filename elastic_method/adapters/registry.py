"""Adapter registry keyed by block family."""

from __future__ import annotations

from typing import Callable

from elastic_method.adapters.base import BlockAdapter

_ADAPTERS: dict[str, BlockAdapter] = {}


def register_block_adapter(factory: Callable[[], BlockAdapter]):
    adapter = factory()
    _ADAPTERS[adapter.family] = adapter
    return factory


def get_block_adapter(family: str) -> BlockAdapter:
    key = str(family).strip().lower()
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        raise KeyError(f"unsupported block family '{family}'") from exc
