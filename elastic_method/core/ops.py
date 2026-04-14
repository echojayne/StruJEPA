"""Low-level masked attention and FFN helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from elastic_method.core.module_utils import apply_layer_norm


def masked_linear(
    x: torch.Tensor,
    linear: nn.Linear,
    *,
    active_out_features: int,
    active_in_features: int | None = None,
) -> torch.Tensor:
    in_features = linear.in_features if active_in_features is None else int(active_in_features)
    weight = linear.weight[:active_out_features, :in_features]
    bias = None if linear.bias is None else linear.bias[:active_out_features]
    return F.linear(x[..., :in_features], weight, bias)


def projected_linear(
    x: torch.Tensor,
    linear: nn.Linear,
    *,
    active_in_features: int,
) -> torch.Tensor:
    weight = linear.weight[:, :active_in_features]
    return F.linear(x[..., :active_in_features], weight, linear.bias)


def build_additive_attention_mask(
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    active_heads: int,
    target_len: int,
    source_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    mask: torch.Tensor | None = None
    if attn_mask is not None:
        mask = attn_mask.to(device=device)
        if mask.dtype == torch.bool:
            float_mask = torch.zeros(mask.shape, device=device, dtype=dtype)
            float_mask.masked_fill_(mask, float("-inf"))
            mask = float_mask
        else:
            mask = mask.to(dtype=dtype)
        if mask.ndim == 2:
            mask = mask.view(1, 1, target_len, source_len)
        elif mask.ndim == 3:
            if mask.shape[0] == batch_size * active_heads:
                mask = mask.view(batch_size, active_heads, target_len, source_len)
            elif mask.shape[0] == batch_size:
                mask = mask.view(batch_size, 1, target_len, source_len)
            else:
                raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)}")
        elif mask.ndim != 4:
            raise ValueError(f"unsupported attention mask rank {mask.ndim}")
    if key_padding_mask is not None:
        padding_mask = key_padding_mask.to(device=device)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(dtype=torch.bool)
        additive = torch.zeros((batch_size, 1, 1, source_len), device=device, dtype=dtype)
        additive.masked_fill_(padding_mask.view(batch_size, 1, 1, source_len), float("-inf"))
        mask = additive if mask is None else mask + additive
    return mask


def scaled_dot_product_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


def apply_activation(module_or_fn: nn.Module | Any, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module_or_fn, nn.Module):
        return module_or_fn(x)
    return module_or_fn(x)


def elastic_torch_mha_forward(
    x: torch.Tensor,
    mha: nn.MultiheadAttention,
    *,
    active_heads: int,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    is_causal: bool,
    batch_first: bool,
) -> torch.Tensor:
    if not mha._qkv_same_embed_dim:
        raise NotImplementedError("separate q/k/v projection weights are not supported in the torch adapter")
    if not batch_first:
        x = x.transpose(0, 1)
    batch_size, seq_len, embed_dim = x.shape
    head_dim = mha.head_dim
    active_dim = active_heads * head_dim
    q_weight, k_weight, v_weight = mha.in_proj_weight.split(embed_dim, dim=0)
    if mha.in_proj_bias is not None:
        q_bias, k_bias, v_bias = mha.in_proj_bias.split(embed_dim, dim=0)
    else:
        q_bias = k_bias = v_bias = None

    q = F.linear(x, q_weight[:active_dim, :], None if q_bias is None else q_bias[:active_dim])
    k = F.linear(x, k_weight[:active_dim, :], None if k_bias is None else k_bias[:active_dim])
    v = F.linear(x, v_weight[:active_dim, :], None if v_bias is None else v_bias[:active_dim])

    q = q.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)

    additive_mask = build_additive_attention_mask(
        attn_mask,
        key_padding_mask,
        batch_size=batch_size,
        active_heads=active_heads,
        target_len=seq_len,
        source_len=seq_len,
        device=x.device,
        dtype=x.dtype,
    )
    context = scaled_dot_product_self_attention(
        q,
        k,
        v,
        dropout_p=mha.dropout if mha.training else 0.0,
        attn_mask=additive_mask,
        is_causal=is_causal and additive_mask is None,
    )
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, active_dim)
    output = F.linear(context, mha.out_proj.weight[:, :active_dim], mha.out_proj.bias)
    if not batch_first:
        output = output.transpose(0, 1).contiguous()
    return output


def elastic_qkv_attention_forward(
    x: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor | None,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor | None,
    active_heads: int,
    head_dim: int,
    attn_dropout_p: float,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    q_norm: nn.Module | None = None,
    k_norm: nn.Module | None = None,
    out_norm: nn.Module | None = None,
    out_dropout: nn.Module | None = None,
    head_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    active_dim = active_heads * head_dim
    q = F.linear(x, q_weight[:active_dim, :], None if q_bias is None else q_bias[:active_dim])
    k = F.linear(x, k_weight[:active_dim, :], None if k_bias is None else k_bias[:active_dim])
    v = F.linear(x, v_weight[:active_dim, :], None if v_bias is None else v_bias[:active_dim])

    q = q.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    if q_norm is not None:
        q = apply_layer_norm(q_norm, q)
    if k_norm is not None:
        k = apply_layer_norm(k_norm, k)
    context = scaled_dot_product_self_attention(
        q,
        k,
        v,
        dropout_p=attn_dropout_p,
        attn_mask=attn_mask,
        is_causal=is_causal and attn_mask is None,
    )
    if head_mask is not None:
        if head_mask.ndim == 1:
            context = context * head_mask[:active_heads].view(1, active_heads, 1, 1)
        elif head_mask.ndim == 2:
            context = context * head_mask[:, :active_heads].view(batch_size, active_heads, 1, 1)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, active_dim)
    if out_norm is not None:
        context = apply_layer_norm(out_norm, context)
    output = F.linear(context, out_weight[:, :active_dim], out_bias)
    if out_dropout is not None:
        output = out_dropout(output)
    return output


def elastic_ffn_forward(
    x: torch.Tensor,
    *,
    fc1: nn.Linear,
    fc2: nn.Linear,
    active_ffn_dim: int,
    activation: nn.Module | Any,
    dropout1: nn.Module | None = None,
    norm: nn.Module | None = None,
    dropout2: nn.Module | None = None,
) -> torch.Tensor:
    hidden = F.linear(x, fc1.weight[:active_ffn_dim, :], None if fc1.bias is None else fc1.bias[:active_ffn_dim])
    hidden = apply_activation(activation, hidden)
    if dropout1 is not None:
        hidden = dropout1(hidden)
    if norm is not None:
        hidden = apply_layer_norm(norm, hidden)
    output = F.linear(hidden, fc2.weight[:, :active_ffn_dim], fc2.bias)
    if dropout2 is not None:
        output = dropout2(output)
    return output
