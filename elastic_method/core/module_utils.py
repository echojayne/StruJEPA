"""Utilities for module path resolution and normalized operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, (nn.ModuleList, nn.Sequential, list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"resolved object at path '{path}' is not an nn.Module")
    return current


def apply_layer_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm, nn.Identity):
        return x
    if isinstance(norm, nn.LayerNorm):
        normalized_dim = int(x.shape[-1])
        weight = norm.weight[:normalized_dim] if norm.elementwise_affine else None
        bias = norm.bias[:normalized_dim] if norm.elementwise_affine else None
        return F.layer_norm(x, (normalized_dim,), weight, bias, norm.eps)
    return norm(x)


def ensure_batch_first(tokens: torch.Tensor, *, batch_first: bool) -> torch.Tensor:
    if batch_first:
        return tokens
    return tokens.transpose(0, 1).contiguous()


def standardize_encoder_tokens(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        if not output:
            raise ValueError("cannot standardize empty tuple output")
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unsupported encoder output type: {type(output)!r}")
    if output.ndim != 3:
        raise ValueError(f"encoder output must be rank-3, got shape {tuple(output.shape)}")
    return output


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(value, "last_hidden_state") and isinstance(value.last_hidden_state, torch.Tensor):
        return value.last_hidden_state
    raise TypeError(f"could not extract tensor from value of type {type(value)!r}")


def replace_modules_in_sequence(
    sequence: nn.Module | Sequence[nn.Module],
    modules: list[nn.Module],
) -> list[nn.Module]:
    if isinstance(sequence, nn.ModuleList):
        if len(sequence) != len(modules):
            raise ValueError("replacement module count must match ModuleList length")
        for idx, module in enumerate(modules):
            sequence[idx] = module
        return list(sequence)
    if isinstance(sequence, nn.Sequential):
        if len(sequence) != len(modules):
            raise ValueError("replacement module count must match Sequential length")
        for idx, module in enumerate(modules):
            sequence[idx] = module
        return list(sequence)
    if isinstance(sequence, list):
        sequence[:] = modules
        return list(sequence)
    raise TypeError(f"unsupported sequence container type: {type(sequence)!r}")
