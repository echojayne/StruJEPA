"""Adapter for HuggingFace ViTLayer stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.vit.modeling_vit import ViTLayer

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.ops import elastic_qkv_attention_forward
from elastic_method.core.structures import ElasticStackMetadata


def _normalize_head_mask(head_mask: torch.Tensor | None, *, active_heads: int) -> torch.Tensor | None:
    if head_mask is None:
        return None
    if head_mask.ndim == 4:
        return head_mask[..., :active_heads, :, :].reshape(head_mask.shape[0], active_heads)
    return head_mask


class ElasticHFViTLayer(ElasticBlockBase):
    """Elastic wrapper for a standard HuggingFace ViTLayer block."""

    def __init__(self, layer: ViTLayer, *, layer_index: int, total_layers: int) -> None:
        attention = layer.attention.attention
        max_heads = int(attention.num_attention_heads)
        max_ffn = int(layer.intermediate.dense.out_features)
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=max_heads,
            max_ffn_dim=max_ffn,
        )
        self.layer = layer
        self.head_dim = int(attention.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return hidden_states
        attention = self.layer.attention.attention
        normed = self.layer.layernorm_before(hidden_states)
        attention_output = elastic_qkv_attention_forward(
            normed,
            q_weight=attention.query.weight,
            q_bias=attention.query.bias,
            k_weight=attention.key.weight,
            k_bias=attention.key.bias,
            v_weight=attention.value.weight,
            v_bias=attention.value.bias,
            out_weight=self.layer.attention.output.dense.weight,
            out_bias=self.layer.attention.output.dense.bias,
            active_heads=active_heads,
            head_dim=self.head_dim,
            attn_dropout_p=attention.dropout.p if attention.training else 0.0,
            head_mask=_normalize_head_mask(head_mask, active_heads=active_heads),
            out_dropout=self.layer.attention.output.dropout,
        )
        hidden_states = attention_output + hidden_states
        layer_output = self.layer.layernorm_after(hidden_states)
        layer_output = F.linear(
            layer_output,
            self.layer.intermediate.dense.weight[:active_ffn_dim, :],
            self.layer.intermediate.dense.bias[:active_ffn_dim],
        )
        layer_output = self.layer.intermediate.intermediate_act_fn(layer_output)
        layer_output = F.linear(
            layer_output,
            self.layer.output.dense.weight[:, :active_ffn_dim],
            self.layer.output.dense.bias,
        )
        layer_output = self.layer.output.dropout(layer_output)
        layer_output = layer_output + hidden_states
        self._record_encoder_state(layer_output, batch_first=True)
        return layer_output


@dataclass
class HFViTAdapter(BlockAdapter):
    family: str = "hf_vit"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_layers = list(sequence)
        if not original_layers:
            raise ValueError("hf_vit adapter received an empty stack")
        if not all(isinstance(layer, ViTLayer) for layer in original_layers):
            raise TypeError("hf_vit adapter expects ViTLayer blocks")
        elastic_layers = [
            ElasticHFViTLayer(layer, layer_index=idx, total_layers=len(original_layers))
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
def _register_hf_vit_adapter() -> BlockAdapter:
    return HFViTAdapter()
