"""Execution-time context shared by the elastic wrapper and converted blocks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ElasticRuntimeState:
    width_multiplier: float
    depth_multiplier: float
    selected_layer_indices: tuple[int, ...]
    active_num_heads: int
    active_ffn_dim: int
    return_encoder_state: bool
    last_encoder_state: torch.Tensor | None = None
    completion_module: Any | None = None
    trace_blocks: bool = False
    block_traces: list[dict[str, torch.Tensor | int]] = field(default_factory=list)
    completion_losses: list[torch.Tensor] = field(default_factory=list)


_CURRENT_RUNTIME: ContextVar[ElasticRuntimeState | None] = ContextVar(
    "elastic_method_runtime",
    default=None,
)


def get_runtime_state() -> ElasticRuntimeState | None:
    return _CURRENT_RUNTIME.get()


@contextmanager
def elastic_runtime(state: ElasticRuntimeState):
    token = _CURRENT_RUNTIME.set(state)
    try:
        yield state
    finally:
        _CURRENT_RUNTIME.reset(token)
