"""Adapter for HuggingFace BertLayer stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer

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


class ElasticHFBertLayer(ElasticBlockBase):
    """Elastic wrapper for a standard HuggingFace BertLayer encoder block."""

    def __init__(self, layer: BertLayer, *, layer_index: int, total_layers: int) -> None:
        if bool(layer.is_decoder):
            raise ValueError("hf_bert adapter only supports encoder BertLayer blocks")
        if hasattr(layer, "crossattention"):
            raise ValueError("hf_bert adapter does not support BertLayer cross-attention")
        max_heads = int(layer.attention.self.num_attention_heads)
        max_ffn = int(layer.intermediate.dense.out_features)
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=max_heads,
            max_ffn_dim=max_ffn,
        )
        self.layer = layer
        self.head_dim = int(layer.attention.self.attention_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        cache_position=None,
    ):
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError("hf_bert adapter only supports encoder self-attention")
        if past_key_values is not None or cache_position is not None:
            raise NotImplementedError("hf_bert adapter does not support cache/past_key_values")
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return (hidden_states, None) if output_attentions else (hidden_states,)

        self_attn = self.layer.attention.self
        attention_output = elastic_qkv_attention_forward(
            hidden_states,
            q_weight=self_attn.query.weight,
            q_bias=self_attn.query.bias,
            k_weight=self_attn.key.weight,
            k_bias=self_attn.key.bias,
            v_weight=self_attn.value.weight,
            v_bias=self_attn.value.bias,
            out_weight=self.layer.attention.output.dense.weight,
            out_bias=self.layer.attention.output.dense.bias,
            active_heads=active_heads,
            head_dim=self.head_dim,
            attn_dropout_p=self_attn.dropout.p if self_attn.training else 0.0,
            attn_mask=attention_mask,
            head_mask=_normalize_head_mask(head_mask, active_heads=active_heads),
        )
        attention_output = self.layer.attention.output.dropout(attention_output)
        attention_output = self.layer.attention.output.LayerNorm(attention_output + hidden_states)

        intermediate_output = F.linear(
            attention_output,
            self.layer.intermediate.dense.weight[:active_ffn_dim, :],
            self.layer.intermediate.dense.bias[:active_ffn_dim],
        )
        intermediate_output = self.layer.intermediate.intermediate_act_fn(intermediate_output)
        layer_output = F.linear(
            intermediate_output,
            self.layer.output.dense.weight[:, :active_ffn_dim],
            self.layer.output.dense.bias,
        )
        layer_output = self.layer.output.dropout(layer_output)
        layer_output = self.layer.output.LayerNorm(layer_output + attention_output)
        self._record_encoder_state(layer_output, batch_first=True)
        return (layer_output, None) if output_attentions else (layer_output,)


@dataclass
class HFBertAdapter(BlockAdapter):
    family: str = "hf_bert"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_layers = list(sequence)
        if not original_layers:
            raise ValueError("hf_bert adapter received an empty stack")
        if not all(isinstance(layer, BertLayer) for layer in original_layers):
            raise TypeError("hf_bert adapter expects BertLayer blocks")
        elastic_layers = [
            ElasticHFBertLayer(layer, layer_index=idx, total_layers=len(original_layers))
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
def _register_hf_bert_adapter() -> BlockAdapter:
    return HFBertAdapter()
