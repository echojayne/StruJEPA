from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method.core.structures import ForwardResult


def parse_multiplier_string(raw: str) -> tuple[float, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        return (1.0,)
    return tuple(float(item) for item in values)


def parse_int_string(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        return ()
    return tuple(int(item) for item in values)


def parse_task_spec_string(
    raw: str | None,
    *,
    default_strategy: str = "random",
    default_ratio: float = 0.5,
) -> tuple[tuple[str, float], ...]:
    if raw is None:
        return ((str(default_strategy), float(default_ratio)),)
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        return ((str(default_strategy), float(default_ratio)),)
    parsed = []
    for item in items:
        if ":" in item:
            strategy, ratio = item.split(":", 1)
            parsed.append((strategy.strip(), float(ratio.strip())))
        else:
            parsed.append((item, float(default_ratio)))
    return tuple(parsed)


def _to_real_tensor(value: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(value):
        return torch.view_as_real(value).flatten(start_dim=-2)
    return value


def _ensure_tensor_batch(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (list, tuple)) and batch and torch.is_tensor(batch[0]):
        return torch.stack(list(batch), dim=0)
    raise TypeError(f"Unsupported WiFo batch type: {type(batch)!r}")


def _masked_patch_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = _to_real_tensor(pred)
    target = _to_real_tensor(target)
    patch_error = torch.abs(pred - target).pow(2).mean(dim=-1)
    mask = mask.float().view_as(patch_error)
    return (patch_error * mask).sum() / mask.sum().clamp_min(1.0)


class WiFoStruJEPACallback:
    def __init__(
        self,
        *,
        representation_dim: int,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random",
        task_specs: str | None = None,
        base_seed: int = 0,
        align_masked_only: bool = True,
    ) -> None:
        self.representation_dim = int(representation_dim)
        self.mask_ratio = float(mask_ratio)
        self.mask_strategy = str(mask_strategy)
        self.task_specs = parse_task_spec_string(
            task_specs,
            default_strategy=self.mask_strategy,
            default_ratio=self.mask_ratio,
        )
        self.base_seed = int(base_seed)
        self.align_masked_only = bool(align_masked_only)

    def prepare_batch(self, batch: Any, device: torch.device) -> dict[str, Any]:
        inputs = _ensure_tensor_batch(batch).to(device=device, dtype=torch.float32, non_blocking=True)
        return {
            "model_args": (inputs,),
            "model_kwargs": {},
            "inputs": inputs,
        }

    def expand_task_batches(self, batch: Any, *, epoch: int, batch_index: int) -> list[dict[str, Any]]:
        task_batches: list[dict[str, Any]] = []
        base_seed = self.base_seed * 1_000_000 + int(epoch) * 10_000 + int(batch_index) * 100
        for task_index, (mask_strategy, mask_ratio) in enumerate(self.task_specs):
            task_batches.append(
                {
                    "model_args": batch["model_args"],
                    "model_kwargs": {
                        "mask_ratio": float(mask_ratio),
                        "mask_strategy": str(mask_strategy),
                        "seed": int(base_seed + task_index),
                    },
                    "inputs": batch["inputs"],
                    "task_name": str(mask_strategy),
                    "task_index": int(task_index),
                }
            )
        return task_batches

    def batch_size(self, batch: Any) -> int:
        return int(batch["inputs"].shape[0])

    def _unpack(self, result: ForwardResult) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = result.model_output
        if not isinstance(outputs, tuple) or len(outputs) != 5:
            raise TypeError("WiFoStruJEPACallback expects WiFo outputs: (loss1, loss2, pred, target, mask)")
        loss1, loss2, pred, target, mask = outputs
        return loss1, loss2, pred, target, mask

    def compute_supervised_loss(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        loss1, _, _, _, _ = self._unpack(result)
        return loss1

    def extract_alignment_view(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        _, _, pred, _, mask = self._unpack(result)
        prediction = _to_real_tensor(pred)
        if not self.align_masked_only:
            return prediction
        return prediction * mask.unsqueeze(-1).to(dtype=prediction.dtype)

    def compute_output_alignment_loss(
        self,
        student_result: ForwardResult,
        teacher_result: ForwardResult,
        batch: Any,
    ) -> torch.Tensor:
        _, _, student_pred, _, student_mask = self._unpack(student_result)
        _, _, teacher_pred, _, _ = self._unpack(teacher_result)
        return _masked_patch_mse(student_pred, teacher_pred.detach(), student_mask)

    def extract_representation(self, result: ForwardResult, batch: Any) -> torch.Tensor:
        if result.encoder_state is None:
            raise ValueError("encoder_state is required for representation extraction")
        return result.encoder_state.mean(dim=1)

    def compute_metrics(self, result: ForwardResult, batch: Any) -> dict[str, float]:
        loss1, _, pred, target, mask = self._unpack(result)
        squared_error = (torch.abs(pred - target) ** 2).mean(dim=-1)
        target_energy = (torch.abs(target) ** 2).mean(dim=-1).clamp_min(1e-8)
        mask = mask.to(dtype=squared_error.dtype)
        nmse = ((squared_error * mask).sum(dim=1) / (target_energy * mask).sum(dim=1).clamp_min(1e-8)).mean()
        return {
            "nmse": float(nmse.item()),
            "recon_loss": float(loss1.item()),
        }
