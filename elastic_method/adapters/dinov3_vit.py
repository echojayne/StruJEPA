"""Adapter for DINOv3 ViT SelfAttentionBlock stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.ops import elastic_ffn_forward, ffn_forward_from_weights
from elastic_method.core.runtime import get_runtime_state
from elastic_method.core.structures import ElasticStackMetadata


def _has_path(module: object, path: str) -> bool:
    current = module
    for part in path.split("."):
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    return True


def _is_supported_dinov3_block(block: nn.Module) -> bool:
    common_paths = (
        "norm1",
        "attn",
        "attn.qkv",
        "attn.proj",
        "attn.num_heads",
        "ls1",
        "norm2",
        "mlp",
        "ls2",
        "sample_drop_ratio",
    )
    if not all(_has_path(block, path) for path in common_paths):
        return False
    mlp_paths = ("mlp.fc1", "mlp.fc2", "mlp.act")
    swiglu_paths = ("mlp.w1", "mlp.w2", "mlp.w3")
    return all(_has_path(block, path) for path in mlp_paths) or all(_has_path(block, path) for path in swiglu_paths)


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


def _rope_rotate_half(x: Tensor) -> Tensor:
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rope_rotate_half(x) * sin)


class ElasticDinoV3Block(ElasticBlockBase):
    """Elastic wrapper around DINOv3 SelfAttentionBlock-like modules."""

    def __init__(self, block: nn.Module, *, layer_index: int, total_layers: int) -> None:
        max_heads = int(block.attn.num_heads)
        if hasattr(block.mlp, "fc1") and hasattr(block.mlp, "fc2"):
            max_ffn = int(block.mlp.fc1.out_features)
            self._ffn_kind = "mlp"
        elif hasattr(block.mlp, "w1") and hasattr(block.mlp, "w2") and hasattr(block.mlp, "w3"):
            max_ffn = int(block.mlp.w1.out_features)
            self._ffn_kind = "swiglu"
        else:
            raise TypeError("unsupported DINOv3 FFN module; expected MLP or SwiGLU")
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=max_heads,
            max_ffn_dim=max_ffn,
        )
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.sample_drop_ratio = float(getattr(block, "sample_drop_ratio", 0.0))

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None
        sin, cos = rope
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        return sin, cos

    @staticmethod
    def _apply_rope_to_qk(
        q: Tensor,
        k: Tensor,
        rope: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        if rope is None:
            return q, k

        sin, cos = rope
        q_dtype = q.dtype
        k_dtype = k.dtype
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        if prefix < 0:
            raise ValueError("rope token count exceeded attention token count")
        q_prefix = q[:, :, :prefix, :]
        k_prefix = k[:, :, :prefix, :]
        q = torch.cat((q_prefix, _rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
        k = torch.cat((k_prefix, _rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def _attention_forward(
        self,
        x: Tensor,
        *,
        rope: tuple[Tensor, Tensor] | None,
        active_heads: int,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape
        head_dim = int(self.attn.qkv.in_features // self.attn.num_heads)
        active_dim = int(active_heads * head_dim)
        out_weight = self.attn.proj.weight
        out_bias = self.attn.proj.bias
        runtime = get_runtime_state()
        if runtime is not None and runtime.completion_module is not None:
            qkv_weight, qkv_bias, out_weight, out_bias = runtime.completion_module.complete_dinov3_attention(
                self,
                active_heads=active_heads,
            )
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
            if qkv_bias is not None:
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
            else:
                q_bias = k_bias = v_bias = None
            active_heads = self.max_num_heads
            active_dim = int(active_heads * head_dim)
        else:
            q_weight, k_weight, v_weight = self.attn.qkv.weight.chunk(3, dim=0)
            if self.attn.qkv.bias is not None:
                q_bias, k_bias, v_bias = self.attn.qkv.bias.chunk(3, dim=0)
            else:
                q_bias = k_bias = v_bias = None

        q = F.linear(x, q_weight[:active_dim, :], None if q_bias is None else q_bias[:active_dim])
        k = F.linear(x, k_weight[:active_dim, :], None if k_bias is None else k_bias[:active_dim])
        v = F.linear(x, v_weight[:active_dim, :], None if v_bias is None else v_bias[:active_dim])

        q = q.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
        q, k = self._apply_rope_to_qk(q, k, rope)

        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, active_dim)
        output = F.linear(context, out_weight[:, :active_dim], out_bias)
        return self.attn.proj_drop(output)

    def _ffn_forward(self, x: Tensor, *, active_ffn_dim: int) -> Tensor:
        runtime = get_runtime_state()
        if runtime is not None and runtime.completion_module is not None:
            completed = runtime.completion_module.complete_dinov3_ffn(self, active_ffn_dim=active_ffn_dim)
            if completed[0] == "mlp":
                _, fc1_weight, fc1_bias, fc2_weight, fc2_bias = completed
                return ffn_forward_from_weights(
                    x,
                    fc1_weight=fc1_weight,
                    fc1_bias=fc1_bias,
                    fc2_weight=fc2_weight,
                    fc2_bias=fc2_bias,
                    activation=self.mlp.act,
                    dropout1=_mlp_drop1(self.mlp),
                    dropout2=_mlp_drop2(self.mlp),
                )
            _, w1_weight, w1_bias, w2_weight, w2_bias, w3_weight, w3_bias = completed
            x1 = F.linear(x, w1_weight, w1_bias)
            x2 = F.linear(x, w2_weight, w2_bias)
            hidden = F.silu(x1) * x2
            return F.linear(hidden, w3_weight, w3_bias)
        if self._ffn_kind == "swiglu":
            w1 = self.mlp.w1
            w2 = self.mlp.w2
            w3 = self.mlp.w3
            x1 = F.linear(x, w1.weight[:active_ffn_dim, :], None if w1.bias is None else w1.bias[:active_ffn_dim])
            x2 = F.linear(x, w2.weight[:active_ffn_dim, :], None if w2.bias is None else w2.bias[:active_ffn_dim])
            hidden = F.silu(x1) * x2
            return F.linear(hidden, w3.weight[:, :active_ffn_dim], w3.bias)
        return elastic_ffn_forward(
            x,
            fc1=self.mlp.fc1,
            fc2=self.mlp.fc2,
            active_ffn_dim=active_ffn_dim,
            activation=self.mlp.act,
            dropout1=_mlp_drop1(self.mlp),
            dropout2=_mlp_drop2(self.mlp),
        )

    def _forward_single(self, x: Tensor, *, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return x

        batch_size = int(x.shape[0])
        sample_subset_size = max(int(batch_size * (1.0 - self.sample_drop_ratio)), 1)
        residual_scale_factor = float(batch_size / sample_subset_size)

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = torch.randperm(batch_size, device=x.device)[:sample_subset_size]
            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self._attention_forward(
                self.norm1(x_subset_1),
                rope=rope_subset,
                active_heads=active_heads,
            )
            attn_residual = torch.index_add(
                torch.zeros_like(x),
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )
            x_attn = x + attn_residual

            indices_2 = torch.randperm(batch_size, device=x.device)[:sample_subset_size]
            x_subset_2 = x_attn[indices_2]
            residual_2 = self._ffn_forward(self.norm2(x_subset_2), active_ffn_dim=active_ffn_dim)
            ffn_residual = torch.index_add(
                torch.zeros_like(x_attn),
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
            output = x_attn + ffn_residual
            self._record_block_trace(
                attention_residual=attn_residual,
                ffn_residual=ffn_residual,
                output=output,
                batch_first=True,
            )
            return output

        attn_residual = self.ls1(self._attention_forward(self.norm1(x), rope=rope, active_heads=active_heads))
        x_attn = x + attn_residual
        ffn_residual = self.ls2(self._ffn_forward(self.norm2(x_attn), active_ffn_dim=active_ffn_dim))
        output = x_attn + ffn_residual
        self._record_block_trace(
            attention_residual=attn_residual,
            ffn_residual=ffn_residual,
            output=output,
            batch_first=True,
        )
        return output

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            output = self._forward_single(x_or_x_list, rope=rope_or_rope_list)
            self._record_encoder_state(output, batch_first=True)
            return output
        if isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in x_or_x_list]
            outputs = [
                self._forward_single(x, rope=rope)
                for x, rope in zip(x_or_x_list, rope_or_rope_list)
            ]
            if len(outputs) == 1:
                self._record_encoder_state(outputs[0], batch_first=True)
            return outputs
        raise TypeError(f"unsupported DINOv3 block input type: {type(x_or_x_list)!r}")


@dataclass
class DinoV3VitAdapter(BlockAdapter):
    family: str = "dinov3_vit"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        sequence = resolve_stack_sequence(target)
        original_blocks = list(sequence)
        if not original_blocks:
            raise ValueError("dinov3_vit adapter received an empty stack")
        if not all(_is_supported_dinov3_block(block) for block in original_blocks):
            raise TypeError("dinov3_vit adapter expects DINOv3 SelfAttentionBlock-style modules")
        elastic_blocks = [
            ElasticDinoV3Block(block, layer_index=idx, total_layers=len(original_blocks))
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
def _register_dinov3_vit_adapter() -> BlockAdapter:
    return DinoV3VitAdapter()
