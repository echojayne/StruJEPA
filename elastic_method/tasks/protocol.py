"""Task callback protocol for the isolated alignment trainer."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from elastic_method.core.structures import ForwardResult


class TaskCallback(Protocol):
    representation_dim: int

    def prepare_batch(self, batch: Any, device: torch.device) -> Any: ...

    def batch_size(self, batch: Any) -> int: ...

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor: ...

    def extract_alignment_view(self, result: ForwardResult, batch: Any) -> torch.Tensor: ...

    def extract_representation(self, result: ForwardResult, batch: Any) -> torch.Tensor: ...

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]: ...
