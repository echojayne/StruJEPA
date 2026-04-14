from __future__ import annotations

import unittest

import torch
from timm.models.vision_transformer import Block
from torch import nn

from elastic_method import ElasticizationSpec, elasticize_model


class ToyTimmVitModel(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(x)
        return self.head(encoded.mean(dim=1))


class TimmVitElasticizationTest(unittest.TestCase):
    def test_full_view_matches_original(self) -> None:
        torch.manual_seed(11)
        model = ToyTimmVitModel()
        elastic = elasticize_model(
            model,
            ElasticizationSpec(
                stack_path="blocks",
                block_family="timm_vit",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
            ),
        )
        inputs = torch.randn(2, 6, 16)
        model.eval()
        elastic.eval()
        with torch.inference_mode():
            baseline = model(inputs)
            wrapped = elastic(inputs, width_multiplier=1.0, depth_multiplier=1.0, return_encoder_state=True)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output, atol=1e-6, rtol=1e-6))
        self.assertEqual(wrapped.encoder_state.shape, (2, 6, 16))


if __name__ == "__main__":
    unittest.main()
