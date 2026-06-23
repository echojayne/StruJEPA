from __future__ import annotations

import unittest

import torch
from torch import nn

from elastic_method import ElasticizationSpec, elasticize_model
from elastic_method.core.subnet import select_depth_indices


class ToyTorchEncoderModel(nn.Module):
    def __init__(self, dim: int = 16, depth: int = 3, heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.head(encoded.mean(dim=1))


class TorchEncoderElasticizationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ToyTorchEncoderModel()
        self.inputs = torch.randn(2, 5, 16)
        self.elastic = elasticize_model(
            self.model,
            ElasticizationSpec(
                stack_path="encoder",
                block_family="torch_encoder",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
                width_only_epochs=1,
            ),
        )

    def test_full_view_matches_original(self) -> None:
        self.model.eval()
        self.elastic.eval()
        with torch.inference_mode():
            baseline = self.model(self.inputs)
            wrapped = self.elastic(self.inputs, width_multiplier=1.0, depth_multiplier=1.0)
        self.assertTrue(torch.allclose(baseline, wrapped.model_output, atol=1e-6, rtol=1e-6))

    def test_forward_result_contains_standardized_encoder_state(self) -> None:
        self.elastic.eval()
        with torch.inference_mode():
            result = self.elastic(
                self.inputs,
                width_multiplier=0.5,
                depth_multiplier=0.5,
                return_encoder_state=True,
            )
        self.assertEqual(result.encoder_state.shape, (2, 5, 16))
        self.assertEqual(result.structure_mask.active_num_heads, 2)
        self.assertEqual(result.structure_mask.active_ffn_dim, 32)
        self.assertEqual(result.structure_mask.selected_layer_indices, (0, 2))

    def test_depth_indices_are_uniform(self) -> None:
        self.assertEqual(select_depth_indices(total_layers=4, active_layers=2), [1, 3])


if __name__ == "__main__":
    unittest.main()
