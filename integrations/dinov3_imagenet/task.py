"""Task adapter for DINOv3 ImageNet classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from elastic_method.core.structures import ForwardResult


class ImageNetClassificationTaskAdapter:
    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        self.label_smoothing = float(label_smoothing)

    @staticmethod
    def _logits(result: ForwardResult) -> torch.Tensor:
        model_output = result.model_output
        if not isinstance(model_output, dict) or "logits" not in model_output:
            raise TypeError("ImageNetClassificationTaskAdapter expects a dict model_output with 'logits'")
        return model_output["logits"]

    def prepare_batch(self, batch: Any, device: torch.device) -> dict[str, Any]:
        images, targets = batch
        return {
            "model_args": (images.to(device, non_blocking=True),),
            "model_kwargs": {},
            "targets": targets.to(device, non_blocking=True),
        }

    def batch_size(self, batch: Any) -> int:
        return int(batch["targets"].shape[0])

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        return F.cross_entropy(
            self._logits(result),
            batch["targets"],
            label_smoothing=self.label_smoothing,
        )

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]:
        logits = self._logits(result)
        targets = batch["targets"]
        maxk = min(5, int(logits.shape[1]))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        top1 = float(correct[:1].reshape(-1).float().mean().item() * 100.0)
        top5 = float(correct[:maxk].any(dim=0).float().mean().item() * 100.0)
        return {"top1": top1, "top5": top5}
