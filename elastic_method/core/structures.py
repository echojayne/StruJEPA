"""Shared data structures for the isolated elastic-method framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ElasticSubnet:
    """A single width-depth subnet operating point."""

    width_multiplier: float
    depth_multiplier: float


@dataclass(frozen=True)
class ElasticizationSpec:
    """Configuration for locating and elasticizing a standard encoder stack."""

    stack_path: str
    block_family: str
    width_multipliers: tuple[float, ...]
    depth_multipliers: tuple[float, ...]
    width_only_epochs: int = 0


@dataclass(frozen=True)
class ElasticStackMetadata:
    """Static metadata needed to interpret a width-depth subnet."""

    family: str
    total_layers: int
    max_num_heads: int
    max_ffn_dim: int


@dataclass(frozen=True)
class StructureMaskDescriptor:
    """Discrete structural mask description for the current subnet."""

    width_multiplier: float
    depth_multiplier: float
    total_layers: int
    selected_layer_indices: tuple[int, ...]
    active_num_heads: int
    active_ffn_dim: int


@dataclass
class ForwardResult:
    """Standardized wrapper around model outputs and elastic stack state."""

    model_output: Any
    encoder_state: torch.Tensor | None
    structure_mask: StructureMaskDescriptor
    aux: dict[str, Any] = field(default_factory=dict)
