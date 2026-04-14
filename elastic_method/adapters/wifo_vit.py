"""Adapter for the upstream WiFo encoder Block family."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.ops import elastic_ffn_forward, elastic_qkv_attention_forward
from elastic_method.core.structures import ElasticStackMetadata


def _is_supported_wifo_block(block: nn.Module) -> bool:
    required_paths = (
        "norm1",
        "attn",
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.proj",
        "attn.num_heads",
        "mlp",
        "mlp.fc1",
        "mlp.fc2",
        "mlp.act",
        "drop_path",
        "norm2",
    )
    current: object
    for path in required_paths:
        current = block
        for part in path.split("."):
            if not hasattr(current, part):
                return False
            current = getattr(current, part)
    return True


def _mlp_drop1(mlp: nn.Module):
    if hasattr(mlp, "drop1"):
        return mlp.drop1
    if hasattr(mlp, "drop"):
        return mlp.drop
    return None


def _mlp_drop2(mlp: nn.Module):
    if hasattr(mlp, "drop2"):
        return mlp.drop2
    if hasattr(mlp, "drop"):
        return mlp.drop
    return None


def _mlp_norm(mlp: nn.Module):
    return getattr(mlp, "norm", None)


class ElasticWifoVitBlock(ElasticBlockBase):
    """Elastic wrapper around the upstream WiFo transformer block."""

    def __init__(self, block: nn.Module, *, layer_index: int, total_layers: int) -> None:
        max_heads = int(block.attn.num_heads)
        max_ffn = int(block.mlp.fc1.out_features)
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=max_heads,
            max_ffn_dim=max_ffn,
        )
        self.norm1 = block.norm1
        self.attn = block.attn
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp

    def forward(self, x):
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return x
        head_dim = int(self.attn.q.out_features // self.attn.num_heads)
        attn_output = elastic_qkv_attention_forward(
            self.norm1(x),
            q_weight=self.attn.q.weight,
            q_bias=self.attn.q.bias,
            k_weight=self.attn.k.weight,
            k_bias=self.attn.k.bias,
            v_weight=self.attn.v.weight,
            v_bias=self.attn.v.bias,
            out_weight=self.attn.proj.weight,
            out_bias=self.attn.proj.bias,
            active_heads=active_heads,
            head_dim=head_dim,
            attn_dropout_p=0.0,
            out_dropout=self.attn.proj_drop,
        )
        x = x + self.drop_path(attn_output)
        ffn_output = elastic_ffn_forward(
            self.norm2(x),
            fc1=self.mlp.fc1,
            fc2=self.mlp.fc2,
            active_ffn_dim=active_ffn_dim,
            activation=self.mlp.act,
            dropout1=_mlp_drop1(self.mlp),
            norm=_mlp_norm(self.mlp),
            dropout2=_mlp_drop2(self.mlp),
        )
        x = x + self.drop_path(ffn_output)
        self._record_encoder_state(x, batch_first=True)
        return x


@dataclass
class WifoVitAdapter(BlockAdapter):
    family: str = "wifo_vit"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_blocks = list(sequence)
        if not original_blocks:
            raise ValueError("wifo_vit adapter received an empty stack")
        if not all(_is_supported_wifo_block(block) for block in original_blocks):
            raise TypeError("wifo_vit adapter expects WiFo-style transformer blocks")
        elastic_blocks = [
            ElasticWifoVitBlock(block, layer_index=idx, total_layers=len(original_blocks))
            for idx, block in enumerate(original_blocks)
        ]
        _, replaced = replace_stack_blocks(target, elastic_blocks)
        metadata = ElasticStackMetadata(
            family=self.family,
            total_layers=len(replaced),
            max_num_heads=elastic_blocks[0].max_num_heads,
            max_ffn_dim=elastic_blocks[0].max_ffn_dim,
        )
        return ElasticizedStackHandle(metadata=metadata, blocks=tuple(replaced))


@register_block_adapter
def _register_wifo_vit_adapter() -> BlockAdapter:
    return WifoVitAdapter()
