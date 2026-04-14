from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from elastic_method import ElasticizationSpec, MethodConfig, elasticize_model
from elastic_method.method.trainer import AlignmentTrainer
from elastic_method.tasks.regression import MeanPooledRegressionCallback


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, length: int = 24, *, seq_len: int = 4, dim: int = 12) -> None:
        self.inputs = torch.randn(length, seq_len, dim)
        self.targets = self.inputs.mean(dim=(1, 2), keepdim=True)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}


class TinyTorchEncoderModel(nn.Module):
    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=3,
            dim_feedforward=dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.head(encoded.mean(dim=1))


class TrainerSmokeTest(unittest.TestCase):
    def test_smoke_training_runs(self) -> None:
        torch.manual_seed(23)
        model = TinyTorchEncoderModel()
        elastic = elasticize_model(
            model,
            ElasticizationSpec(
                stack_path="encoder",
                block_family="torch_encoder",
                width_multipliers=(1.0, 0.5),
                depth_multipliers=(1.0, 0.5),
                width_only_epochs=1,
            ),
        )
        callback = MeanPooledRegressionCallback(representation_dim=12)
        trainer = AlignmentTrainer(
            elastic,
            callback,
            spec=elastic.spec,
            config=MethodConfig(
                use_ema_full_view=True,
                enable_output_alignment=True,
                enable_repr_alignment=True,
                lambda_output=0.5,
                lambda_repr=0.05,
            ),
            device="cpu",
        )
        loader = DataLoader(TinyDataset(), batch_size=6, shuffle=False)
        history = trainer.fit(loader, epochs=2)
        self.assertEqual(len(history), 2)
        self.assertIn("train_loss", history[-1])
        self.assertIn("train_output_alignment_loss", history[-1])
        self.assertIn("train_repr_alignment_loss", history[-1])
        self.assertGreaterEqual(history[-1]["train_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
