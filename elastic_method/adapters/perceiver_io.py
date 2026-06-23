"""Adapter for BeamFormer-style lucidrains Perceiver IO latent blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.common import ElasticBlockBase, get_target_module, replace_stack_blocks
from elastic_method.core.ops import qkv_attention_forward_from_weights
from elastic_method.adapters.registry import register_block_adapter
from elastic_method.core.runtime import get_runtime_state
from elastic_method.core.structures import ElasticStackMetadata


def _require_module_list(block: nn.Module) -> nn.ModuleList:
    if not isinstance(block, nn.ModuleList) or len(block) != 2:
        raise TypeError("perceiver_io adapter expects each latent block to be ModuleList([attn, ffn])")
    return block


def _feedforward_linears(feedforward: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    net = getattr(feedforward, "net", None)
    if not isinstance(net, nn.Sequential) or len(net) < 3:
        raise TypeError("perceiver_io adapter expects FeedForward.net = [Linear, GEGLU, Linear]")
    fc1 = net[0]
    fc2 = net[2]
    if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
        raise TypeError("perceiver_io FeedForward linears are not nn.Linear")
    if int(fc1.out_features) != int(fc2.in_features) * 2:
        raise ValueError("perceiver_io GEGLU fc1 rows must equal 2 * fc2 input width")
    return fc1, fc2


def _geglu_forward(hidden: torch.Tensor) -> torch.Tensor:
    values, gates = hidden.chunk(2, dim=-1)
    return values * F.gelu(gates)


class ElasticPerceiverIOLatentBlock(ElasticBlockBase):
    """Elastic wrapper for one Perceiver IO latent self-attention/FFN block."""

    def __init__(self, block: nn.Module, *, layer_index: int, total_layers: int) -> None:
        module_list = _require_module_list(block)
        attention_prenorm = module_list[0]
        ffn_prenorm = module_list[1]
        attention = getattr(attention_prenorm, "fn", None)
        feedforward = getattr(ffn_prenorm, "fn", None)
        if attention is None or feedforward is None:
            raise TypeError("perceiver_io adapter expects PreNorm wrappers with .fn")
        if not all(hasattr(attention, name) for name in ("to_q", "to_kv", "to_out", "heads", "scale")):
            raise TypeError("perceiver_io attention must expose to_q, to_kv, to_out, heads, and scale")
        if not isinstance(attention.to_q, nn.Linear) or not isinstance(attention.to_kv, nn.Linear):
            raise TypeError("perceiver_io attention projections must be nn.Linear")
        if not isinstance(attention.to_out, nn.Linear):
            raise TypeError("perceiver_io attention output projection must be nn.Linear")
        fc1, fc2 = _feedforward_linears(feedforward)
        heads = int(attention.heads)
        inner_dim = int(attention.to_q.out_features)
        if inner_dim % heads != 0:
            raise ValueError("perceiver_io attention inner dim must be divisible by heads")
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=heads,
            max_ffn_dim=int(fc2.in_features),
        )
        self.attention_prenorm = attention_prenorm
        self.ffn_prenorm = ffn_prenorm
        self.attention = attention
        self.feedforward = feedforward
        self.fc1 = fc1
        self.fc2 = fc2
        self.head_dim = inner_dim // heads
        self._last_attention_residual: torch.Tensor | None = None

    def __iter__(self):
        """Match upstream PerceiverIO.forward, which unpacks each layer."""

        yield self._attention_residual_forward
        yield self._ffn_residual_forward

    def _attention(self, x: torch.Tensor, *, active_heads: int) -> torch.Tensor:
        runtime = get_runtime_state()
        if runtime is not None and runtime.completion_module is not None:
            q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias = (
                runtime.completion_module.complete_perceiver_io_attention(
                    self,
                    active_heads=active_heads,
                )
            )
            return qkv_attention_forward_from_weights(
                x,
                q_weight=q_weight,
                q_bias=q_bias,
                k_weight=k_weight,
                k_bias=k_bias,
                v_weight=v_weight,
                v_bias=v_bias,
                out_weight=out_weight,
                out_bias=out_bias,
                num_heads=self.max_num_heads,
                head_dim=self.head_dim,
                attn_dropout_p=0.0,
            )
        active_dim = int(active_heads) * int(self.head_dim)
        q = F.linear(x, self.attention.to_q.weight[:active_dim, :], None)
        k_weight, v_weight = self.attention.to_kv.weight.chunk(2, dim=0)
        k = F.linear(x, k_weight[:active_dim, :], None)
        v = F.linear(x, v_weight[:active_dim, :], None)
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=int(active_heads)),
            (q, k, v),
        )
        sim = einsum("b i d, b j d -> b i j", q, k) * float(self.attention.scale)
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n h d", h=int(active_heads))
        head_mask = getattr(self, "_dynabert_head_mask", None)
        if isinstance(head_mask, torch.Tensor):
            active_mask = head_mask[: int(active_heads)].to(device=out.device, dtype=out.dtype)
            out = out * active_mask.view(1, 1, int(active_heads), 1)
        out = rearrange(out, "b n h d -> b n (h d)")
        return F.linear(out, self.attention.to_out.weight[:, :active_dim], self.attention.to_out.bias)

    def _ffn(self, x: torch.Tensor, *, active_ffn_dim: int) -> torch.Tensor:
        runtime = get_runtime_state()
        if runtime is not None and runtime.completion_module is not None:
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = runtime.completion_module.complete_perceiver_io_ffn(
                self,
                active_ffn_dim=active_ffn_dim,
            )
            hidden = F.linear(x, fc1_weight, fc1_bias)
            hidden = _geglu_forward(hidden)
            return F.linear(hidden, fc2_weight, fc2_bias)
        active_ffn_dim = max(1, min(int(active_ffn_dim), int(self.max_ffn_dim)))
        value_rows = torch.arange(active_ffn_dim, device=x.device)
        gate_rows = value_rows + int(self.max_ffn_dim)
        row_index = torch.cat((value_rows, gate_rows), dim=0)
        hidden = F.linear(
            x,
            self.fc1.weight.index_select(0, row_index),
            None if self.fc1.bias is None else self.fc1.bias.index_select(0, row_index),
        )
        hidden = _geglu_forward(hidden)
        return F.linear(hidden, self.fc2.weight[:, :active_ffn_dim], self.fc2.bias)

    def _attention_residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            self._last_attention_residual = None
            return torch.zeros_like(x)
        attn_input = self.attention_prenorm.norm(x)
        attn_residual = self._attention(attn_input, active_heads=active_heads)
        self._last_attention_residual = attn_residual
        return attn_residual

    def _ffn_residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        is_active, _active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return torch.zeros_like(x)
        ffn_input = self.ffn_prenorm.norm(x)
        ffn_residual = self._ffn(ffn_input, active_ffn_dim=active_ffn_dim)
        output = x + ffn_residual
        attention_residual = self._last_attention_residual
        self._last_attention_residual = None
        if attention_residual is not None:
            self._record_block_trace(attention_residual=attention_residual, ffn_residual=ffn_residual, output=output)
        self._record_encoder_state(output)
        return ffn_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_residual = self._attention_residual_forward(x)
        x = x + attn_residual
        ffn_residual = self._ffn_residual_forward(x)
        x = x + ffn_residual
        return x


@dataclass
class PerceiverIOAdapter(BlockAdapter):
    family: str = "perceiver_io"

    def elasticize(self, model: nn.Module, *, stack_path: str) -> ElasticizedStackHandle:
        target = get_target_module(model, stack_path)
        if not isinstance(target, nn.ModuleList):
            raise TypeError("perceiver_io adapter stack_path must point to PerceiverIO.layers ModuleList")
        original_layers = list(target)
        if not original_layers:
            raise ValueError("perceiver_io adapter received an empty stack")
        elastic_layers = [
            ElasticPerceiverIOLatentBlock(layer, layer_index=idx, total_layers=len(original_layers))
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
def _register_perceiver_io_adapter() -> BlockAdapter:
    return PerceiverIOAdapter()
