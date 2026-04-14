"""Subnet helpers shared by the elastic model wrapper and trainer."""

from __future__ import annotations

import math
from typing import Iterable

from elastic_method.core.structures import ElasticSubnet


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def resolve_active_heads(*, max_heads: int, width_multiplier: float) -> int:
    return _clamp_int(int(round(max_heads * float(width_multiplier))), 1, max_heads)


def resolve_active_ffn(*, max_ffn_dim: int, width_multiplier: float) -> int:
    return _clamp_int(int(round(max_ffn_dim * float(width_multiplier))), 1, max_ffn_dim)


def resolve_active_layers(*, max_layers: int, depth_multiplier: float) -> int:
    return _clamp_int(int(round(max_layers * float(depth_multiplier))), 1, max_layers)


def select_depth_indices(*, total_layers: int, active_layers: int) -> list[int]:
    if active_layers >= total_layers:
        return list(range(total_layers))
    return [
        max(0, math.floor((layer_idx + 1) * total_layers / active_layers) - 1)
        for layer_idx in range(active_layers)
    ]


def resolve_multiplier_list(values: Iterable[float] | None) -> list[float]:
    ordered = [float(value) for value in values] if values is not None else []
    ordered.append(1.0)
    clipped = [min(1.0, max(float(value), 0.0)) for value in ordered if float(value) > 0.0]
    return sorted(set(clipped), reverse=True)


def dedupe_subnets(
    width_multipliers: Iterable[float],
    depth_multipliers: Iterable[float],
) -> list[ElasticSubnet]:
    ordered: list[ElasticSubnet] = []
    seen: set[tuple[float, float]] = set()
    for depth_multiplier in depth_multipliers:
        for width_multiplier in width_multipliers:
            key = (float(width_multiplier), float(depth_multiplier))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(ElasticSubnet(*key))
    return ordered
