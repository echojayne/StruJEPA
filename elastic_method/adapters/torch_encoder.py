"""Adapter for torch.nn.TransformerEncoderLayer family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.ops import elastic_ffn_forward, elastic_torch_mha_forward
from elastic_method.core.structures import ElasticStackMetadata


class ElasticTorchEncoderLayer(ElasticBlockBase):
    """Elastic wrapper for TransformerEncoderLayer with compute-width semantics."""

    def __init__(self, layer: nn.TransformerEncoderLayer, *, layer_index: int, total_layers: int) -> None:
        if not isinstance(layer.self_attn, nn.MultiheadAttention):
            raise TypeError("torch encoder adapter requires MultiheadAttention self_attn")
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=int(layer.self_attn.num_heads),
            max_ffn_dim=int(layer.linear1.out_features),
        )
        self.self_attn = layer.self_attn
        self.linear1 = layer.linear1
        self.dropout = layer.dropout
        self.linear2 = layer.linear2
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.activation = layer.activation
        self.norm_first = bool(layer.norm_first)
        self.batch_first = bool(layer.self_attn.batch_first)

    def _self_attention(
        self,
        src: torch.Tensor,
        *,
        active_heads: int,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        return elastic_torch_mha_forward(
            src,
            self.self_attn,
            active_heads=active_heads,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            batch_first=self.batch_first,
        )

    def _ffn(self, x: torch.Tensor, *, active_ffn_dim: int) -> torch.Tensor:
        return elastic_ffn_forward(
            x,
            fc1=self.linear1,
            fc2=self.linear2,
            active_ffn_dim=active_ffn_dim,
            activation=self.activation,
            dropout1=self.dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return src
        x = src
        if self.norm_first:
            x = x + self.dropout1(
                self._self_attention(
                    self.norm1(x),
                    active_heads=active_heads,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = x + self.dropout2(self._ffn(self.norm2(x), active_ffn_dim=active_ffn_dim))
        else:
            x = self.norm1(
                x
                + self.dropout1(
                    self._self_attention(
                        x,
                        active_heads=active_heads,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        is_causal=is_causal,
                    )
                )
            )
            x = self.norm2(x + self.dropout2(self._ffn(x, active_ffn_dim=active_ffn_dim)))
        self._record_encoder_state(x, batch_first=self.batch_first)
        return x


@dataclass
class TorchEncoderAdapter(BlockAdapter):
    family: str = "torch_encoder"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_layers = list(sequence)
        if not original_layers:
            raise ValueError("torch encoder adapter received an empty stack")
        if not all(isinstance(layer, nn.TransformerEncoderLayer) for layer in original_layers):
            raise TypeError("torch encoder adapter expects TransformerEncoderLayer blocks")
        elastic_layers = [
            ElasticTorchEncoderLayer(layer, layer_index=idx, total_layers=len(original_layers))
            for idx, layer in enumerate(original_layers)
        ]
        _, replaced = replace_stack_blocks(target, elastic_layers)
        metadata = ElasticStackMetadata(
            family=self.family,
            total_layers=len(replaced),
            max_num_heads=elastic_layers[0].max_num_heads,
            max_ffn_dim=elastic_layers[0].max_ffn_dim,
        )
        return ElasticizedStackHandle(metadata=metadata, blocks=tuple(replaced))


@register_block_adapter
def _register_torch_encoder_adapter() -> BlockAdapter:
    return TorchEncoderAdapter()
