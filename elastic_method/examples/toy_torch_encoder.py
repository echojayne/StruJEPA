"""Toy smoke example for the isolated elastic framework."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from elastic_method import ElasticizationSpec, MethodConfig, elasticize_model
from elastic_method.method.trainer import AlignmentTrainer
from elastic_method.tasks.regression import MeanPooledRegressionCallback


class ToySequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, length: int = 32, *, seq_len: int = 6, dim: int = 16) -> None:
        self.inputs = torch.randn(length, seq_len, dim)
        self.targets = self.inputs.mean(dim=(1, 2), keepdim=True)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}


class ToyTorchEncoderRegressor(nn.Module):
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
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.head(encoded.mean(dim=1))


def main() -> None:
    model = ToyTorchEncoderRegressor()
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
    callback = MeanPooledRegressionCallback(representation_dim=16)
    trainer = AlignmentTrainer(
        elastic,
        callback,
        spec=elastic.spec,
        config=MethodConfig(use_ema_full_view=True, enable_output_alignment=True, enable_repr_alignment=True),
        device="cpu",
    )
    dataset = ToySequenceDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    history = trainer.fit(loader, epochs=2)
    print(history[-1])


if __name__ == "__main__":
    main()
