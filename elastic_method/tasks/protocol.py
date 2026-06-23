"""Task adapter protocol for StruJEPA training."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from elastic_method.core.structures import ForwardResult


class TaskAdapter(Protocol):
    def prepare_batch(self, batch: Any, device: torch.device) -> Any: ...

    def batch_size(self, batch: Any) -> int: ...

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor: ...

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]: ...
