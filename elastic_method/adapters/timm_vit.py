"""Adapter for timm VisionTransformer Block family."""

from __future__ import annotations

from dataclasses import dataclass

from timm.models.vision_transformer import Block
from torch import nn

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.ops import elastic_ffn_forward, elastic_qkv_attention_forward
from elastic_method.core.structures import ElasticStackMetadata


class ElasticTimmVitBlock(ElasticBlockBase):
    """Elastic wrapper around timm.models.vision_transformer.Block."""

    def __init__(self, block: Block, *, layer_index: int, total_layers: int) -> None:
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
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

    def forward(
        self,
        x,
        attn_mask=None,
        is_causal: bool = False,
    ):
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return x
        active_dim = active_heads * self.attn.head_dim
        q_weight, k_weight, v_weight = self.attn.qkv.weight.split(self.attn.attn_dim, dim=0)
        if self.attn.qkv.bias is not None:
            q_bias, k_bias, v_bias = self.attn.qkv.bias.split(self.attn.attn_dim, dim=0)
        else:
            q_bias = k_bias = v_bias = None
        attn_output = elastic_qkv_attention_forward(
            self.norm1(x),
            q_weight=q_weight,
            q_bias=q_bias,
            k_weight=k_weight,
            k_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            out_weight=self.attn.proj.weight,
            out_bias=self.attn.proj.bias,
            active_heads=active_heads,
            head_dim=self.attn.head_dim,
            attn_dropout_p=self.attn.attn_drop.p if self.attn.training else 0.0,
            attn_mask=attn_mask,
            is_causal=is_causal,
            q_norm=self.attn.q_norm,
            k_norm=self.attn.k_norm,
            out_norm=self.attn.norm,
            out_dropout=self.attn.proj_drop,
        )
        x = x + self.drop_path1(self.ls1(attn_output))
        ffn_output = elastic_ffn_forward(
            self.norm2(x),
            fc1=self.mlp.fc1,
            fc2=self.mlp.fc2,
            active_ffn_dim=active_ffn_dim,
            activation=self.mlp.act,
            dropout1=self.mlp.drop1,
            norm=self.mlp.norm,
            dropout2=self.mlp.drop2,
        )
        x = x + self.drop_path2(self.ls2(ffn_output))
        self._record_encoder_state(x, batch_first=True)
        return x


@dataclass
class TimmVitAdapter(BlockAdapter):
    family: str = "timm_vit"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_blocks = list(sequence)
        if not original_blocks:
            raise ValueError("timm_vit adapter received an empty stack")
        if not all(isinstance(block, Block) for block in original_blocks):
            raise TypeError("timm_vit adapter expects timm Block instances")
        elastic_blocks = [
            ElasticTimmVitBlock(block, layer_index=idx, total_layers=len(original_blocks))
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
def _register_timm_vit_adapter() -> BlockAdapter:
    return TimmVitAdapter()
