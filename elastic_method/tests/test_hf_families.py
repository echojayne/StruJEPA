from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import BertConfig, ViTConfig
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer

from elastic_method import ElasticizationSpec, elasticize_model


class ToyBertStackModel(nn.Module):
    def __init__(self, hidden_size: int = 16, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        cfg = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=heads,
            intermediate_size=hidden_size * 2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        cfg._attn_implementation = "eager"
        self.layers = nn.ModuleList([BertLayer(cfg) for _ in range(depth)])
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return self.head(hidden.mean(dim=1))


class ToyHFViTStackModel(nn.Module):
    def __init__(self, hidden_size: int = 16, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        cfg = ViTConfig(
            hidden_size=hidden_size,
            num_attention_heads=heads,
            intermediate_size=hidden_size * 2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        cfg._attn_implementation = "eager"
        self.layers = nn.ModuleList([ViTLayer(cfg) for _ in range(depth)])
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden.mean(dim=1))


class HFFamiliesElasticizationTest(unittest.TestCase):
    def test_hf_bert_stack_matches_full_view(self) -> None:
        torch.manual_seed(13)
        model = ToyBertStackModel()
        elastic = elasticize_model(
            model,
            ElasticizationSpec(
                stack_path="layers",
                block_family="hf_bert",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
            ),
        )
        inputs = torch.randn(2, 5, 16)
        model.eval()
        elastic.eval()
        with torch.inference_mode():
            baseline = model(inputs)
            wrapped = elastic(inputs, width_multiplier=1.0, depth_multiplier=1.0, return_encoder_state=True)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output, atol=1e-6, rtol=1e-6))
        self.assertEqual(wrapped.encoder_state.shape, (2, 5, 16))

    def test_hf_vit_stack_matches_full_view(self) -> None:
        torch.manual_seed(17)
        model = ToyHFViTStackModel()
        elastic = elasticize_model(
            model,
            ElasticizationSpec(
                stack_path="layers",
                block_family="hf_vit",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
            ),
        )
        inputs = torch.randn(2, 5, 16)
        model.eval()
        elastic.eval()
        with torch.inference_mode():
            baseline = model(inputs)
            wrapped = elastic(inputs, width_multiplier=1.0, depth_multiplier=1.0, return_encoder_state=True)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output, atol=1e-6, rtol=1e-6))
        self.assertEqual(wrapped.encoder_state.shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
