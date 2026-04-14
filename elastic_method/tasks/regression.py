"""Minimal regression callback used by examples and smoke tests."""

from __future__ import annotations

from typing import Any

import torch

from elastic_method.core.structures import ForwardResult


class MeanPooledRegressionCallback:
    """Treat the model output as a regression tensor and pool encoder tokens by mean."""

    def __init__(self, *, representation_dim: int) -> None:
        self.representation_dim = int(representation_dim)

    def prepare_batch(self, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        return {"model_args": (inputs,), "model_kwargs": {}, "targets": targets}

    def batch_size(self, batch: Any) -> int:
        return int(batch["targets"].shape[0])

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        return torch.mean((result.model_output - batch["targets"]) ** 2)

    def extract_alignment_view(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        if not isinstance(result.model_output, torch.Tensor):
            raise TypeError("MeanPooledRegressionCallback expects tensor model_output")
        return result.model_output

    def extract_representation(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        if result.encoder_state is None:
            raise ValueError("encoder_state is required for representation extraction")
        return result.encoder_state.mean(dim=1)

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]:
        diff = result.model_output - batch["targets"]
        mse = float(torch.mean(diff**2).item())
        mae = float(torch.mean(torch.abs(diff)).item())
        return {"mse": mse, "mae": mae}
