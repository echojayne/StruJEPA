from __future__ import annotations

import random
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import StruJEPATrainer


class WiFoStruJEPATrainer(StruJEPATrainer):
    def __init__(
        self,
        *args,
        random_subnets_per_batch: int = 0,
        sampling_seed: int = 0,
        validate_every: int = 1,
        log_every_batches: int = 0,
        subnet_sampling_mode: str = "anchor_random",
        objective_mode: str = "full_plus_mean_subnets",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.random_subnets_per_batch = int(random_subnets_per_batch)
        self.sampling_seed = int(sampling_seed)
        self.validate_every = int(validate_every)
        self.log_every_batches = int(log_every_batches)
        self.subnet_sampling_mode = str(subnet_sampling_mode)
        self.objective_mode = str(objective_mode)

    @staticmethod
    def _pick_middle_value(values: list[float]) -> float:
        if not values:
            raise ValueError("values must not be empty")
        return values[len(values) // 2]

    def _sample_subnets(self, all_subnets: list[Any], *, epoch: int, batch_index: int) -> list[Any]:
        if self.subnet_sampling_mode == "all" or len(all_subnets) <= 1:
            return list(all_subnets)

        width_values = sorted({float(subnet.width_multiplier) for subnet in all_subnets}, reverse=True)
        depth_values = sorted({float(subnet.depth_multiplier) for subnet in all_subnets}, reverse=True)
        mid_width = self._pick_middle_value(width_values)
        mid_depth = self._pick_middle_value(depth_values)
        if self.subnet_sampling_mode == "scale_anchors":
            anchor_keys = [
                (width_values[0], depth_values[0]),
                (mid_width, mid_depth),
                (width_values[-1], depth_values[-1]),
            ]
        elif self.subnet_sampling_mode == "anchor_random":
            anchor_keys = [
                (width_values[0], depth_values[0]),
                (mid_width, mid_depth),
                (width_values[-1], depth_values[-1]),
                (width_values[-1], depth_values[0]),
                (width_values[0], depth_values[-1]),
            ]
        else:
            raise ValueError(f"unsupported subnet_sampling_mode '{self.subnet_sampling_mode}'")

        subnet_map = {
            (float(subnet.width_multiplier), float(subnet.depth_multiplier)): subnet for subnet in all_subnets
        }
        selected = []
        seen: set[tuple[float, float]] = set()
        for key in anchor_keys:
            subnet = subnet_map.get(key)
            if subnet is not None and key not in seen:
                selected.append(subnet)
                seen.add(key)

        if self.subnet_sampling_mode == "anchor_random":
            remaining = [
                subnet
                for subnet in all_subnets
                if (float(subnet.width_multiplier), float(subnet.depth_multiplier)) not in seen
            ]
            rng = random.Random(self.sampling_seed + epoch * 10_000 + batch_index)
            rng.shuffle(remaining)
            selected.extend(remaining[: max(0, self.random_subnets_per_batch)])
        return selected

    def _select_subnets_for_batch(
        self,
        all_subnets: list[Any],
        *,
        epoch: int,
        batch_index: int,
    ) -> list[Any]:
        return self._sample_subnets(all_subnets, epoch=epoch, batch_index=batch_index)

    def _expand_task_batches(self, batch: Any, *, epoch: int, batch_index: int) -> list[Any]:
        expand = getattr(self.task_adapter, "expand_task_batches", None)
        if not callable(expand):
            return [batch]
        task_batches = list(expand(batch, epoch=epoch, batch_index=batch_index))
        if not task_batches:
            raise ValueError("task_adapter.expand_task_batches returned no tasks")
        return task_batches

    def _combine_task_losses(
        self,
        full_loss: torch.Tensor,
        subnet_losses: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.objective_mode == "full_plus_mean_subnets":
            return full_loss + (torch.stack(subnet_losses).mean() if subnet_losses else 0.0)
        if self.objective_mode == "mean_all":
            return torch.stack([full_loss, *subnet_losses]).mean()
        raise ValueError(f"unsupported objective_mode '{self.objective_mode}'")

    def fit(
        self,
        train_loader: Any,
        *,
        epochs: int,
        val_loader: Any | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(1, int(epochs) + 1):
            train_metrics = self.run_epoch(train_loader, epoch=epoch, train=True)
            record = {"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}}
            should_validate = (
                val_loader is not None
                and (epoch % max(1, self.validate_every) == 0 or epoch == int(epochs))
            )
            if should_validate:
                val_metrics = self.run_epoch(val_loader, epoch=epoch, train=False)
                record.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(record)
        return history
