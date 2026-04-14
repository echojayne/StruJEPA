"""Common adapter utilities and base elastic block helpers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from elastic_method.core.module_utils import ensure_batch_first, resolve_module_path, replace_modules_in_sequence
from elastic_method.core.runtime import get_runtime_state


class ElasticBlockBase(nn.Module):
    """Base mixin for family-specific elastic blocks."""

    def __init__(self, *, layer_index: int, total_layers: int, max_num_heads: int, max_ffn_dim: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.total_layers = int(total_layers)
        self.max_num_heads = int(max_num_heads)
        self.max_ffn_dim = int(max_ffn_dim)

    def _is_active(self) -> tuple[bool, int, int]:
        runtime = get_runtime_state()
        if runtime is None:
            return True, self.max_num_heads, self.max_ffn_dim
        return (
            self.layer_index in runtime.selected_layer_indices,
            int(runtime.active_num_heads),
            int(runtime.active_ffn_dim),
        )

    def _record_encoder_state(self, tokens: torch.Tensor, *, batch_first: bool = True) -> None:
        runtime = get_runtime_state()
        if runtime is None or not runtime.return_encoder_state:
            return
        runtime.last_encoder_state = ensure_batch_first(tokens, batch_first=batch_first)


def resolve_stack_sequence(target_module: nn.Module) -> nn.Module:
    """Map a supported stack container to the sequence of repeated blocks."""

    if isinstance(target_module, (nn.ModuleList, nn.Sequential)):
        return target_module
    if hasattr(target_module, "layers") and isinstance(target_module.layers, (nn.ModuleList, nn.Sequential)):
        return target_module.layers
    if hasattr(target_module, "layer") and isinstance(target_module.layer, (nn.ModuleList, nn.Sequential)):
        return target_module.layer
    raise TypeError(f"unsupported stack container type: {type(target_module)!r}")


def replace_stack_blocks(target_module: nn.Module, blocks: list[nn.Module]) -> tuple[nn.Module, list[nn.Module]]:
    sequence = resolve_stack_sequence(target_module)
    replaced = replace_modules_in_sequence(sequence, blocks)
    return target_module, replaced


def get_target_module(model: nn.Module, stack_path: str) -> nn.Module:
    target = resolve_module_path(model, stack_path)
    if not isinstance(target, nn.Module):
        raise TypeError(f"target at path '{stack_path}' is not an nn.Module")
    return target
