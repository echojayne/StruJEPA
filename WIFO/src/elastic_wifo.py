from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import ElasticizationSpec, elasticize_model


def _normalize_count_values(raw_values, *, maximum: int, minimum: int = 1) -> tuple[int, ...]:
    values = {int(value) for value in raw_values}
    clipped = {min(maximum, max(minimum, value)) for value in values}
    clipped.add(int(maximum))
    return tuple(sorted(clipped, reverse=True))


def counts_to_multipliers(raw_values, *, maximum: int, minimum: int = 1) -> tuple[float, ...]:
    counts = _normalize_count_values(raw_values, maximum=maximum, minimum=minimum)
    return tuple(float(value) / float(maximum) for value in counts)


def build_headwise_width_multipliers(model, *, active_head_values=None, min_active_heads: int = 1) -> tuple[float, ...]:
    max_heads = int(model.num_heads)
    values = active_head_values if active_head_values is not None else range(min_active_heads, max_heads + 1)
    return counts_to_multipliers(values, maximum=max_heads, minimum=1)


def build_layerwise_depth_multipliers(model, *, active_layer_values=None, min_active_layers: int = 1) -> tuple[float, ...]:
    max_layers = int(model.depth)
    values = active_layer_values if active_layer_values is not None else range(min_active_layers, max_layers + 1)
    return counts_to_multipliers(values, maximum=max_layers, minimum=1)


def elasticize_wifo(
    model,
    *,
    width_multipliers=(1.0, 0.75, 0.5),
    depth_multipliers=(1.0, 0.75, 0.5),
    width_only_epochs=0,
    stack_path="blocks",
    copy_model=True,
):
    spec = ElasticizationSpec(
        stack_path=stack_path,
        block_family="wifo_vit",
        width_multipliers=tuple(float(value) for value in width_multipliers),
        depth_multipliers=tuple(float(value) for value in depth_multipliers),
        width_only_epochs=int(width_only_epochs),
    )
    return elasticize_model(model, spec, copy_model=copy_model)
