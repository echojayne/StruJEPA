"""Shared adapter interfaces and handle types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import nn

from elastic_method.core.structures import ElasticStackMetadata


@dataclass(frozen=True)
class ElasticizedStackHandle:
    metadata: ElasticStackMetadata
    blocks: tuple[nn.Module, ...]


class BlockAdapter(Protocol):
    family: str

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle: ...
