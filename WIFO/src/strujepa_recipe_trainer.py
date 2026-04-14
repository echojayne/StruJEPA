from __future__ import annotations

import json
import random
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elastic_method import AlignmentTrainer


class WiFoStruJEPATrainer(AlignmentTrainer):
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
        if self.subnet_sampling_mode != "anchor_random":
            raise ValueError(f"unsupported subnet_sampling_mode '{self.subnet_sampling_mode}'")

        width_values = sorted({float(subnet.width_multiplier) for subnet in all_subnets}, reverse=True)
        depth_values = sorted({float(subnet.depth_multiplier) for subnet in all_subnets}, reverse=True)
        mid_width = self._pick_middle_value(width_values)
        mid_depth = self._pick_middle_value(depth_values)
        anchor_keys = [
            (width_values[0], depth_values[0]),
            (mid_width, mid_depth),
            (width_values[-1], depth_values[-1]),
            (width_values[-1], depth_values[0]),
            (width_values[0], depth_values[-1]),
        ]
        subnet_map = {
            (float(subnet.width_multiplier), float(subnet.depth_multiplier)): subnet for subnet in all_subnets
        }
        selected = []
        seen: set[tuple[float, float]] = set()
        for key in anchor_keys:
            subnet = subnet_map.get(key)
            if subnet is None or key in seen:
                continue
            selected.append(subnet)
            seen.add(key)

        remaining = [
            subnet
            for subnet in all_subnets
            if (float(subnet.width_multiplier), float(subnet.depth_multiplier)) not in seen
        ]
        rng = random.Random(self.sampling_seed + epoch * 10_000 + batch_index)
        rng.shuffle(remaining)
        for subnet in remaining[: max(0, self.random_subnets_per_batch)]:
            key = (float(subnet.width_multiplier), float(subnet.depth_multiplier))
            if key in seen:
                continue
            selected.append(subnet)
            seen.add(key)
        return selected

    def _expand_task_batches(self, batch: Any, *, epoch: int, batch_index: int) -> list[Any]:
        expand = getattr(self.callback, "expand_task_batches", None)
        if callable(expand):
            task_batches = list(expand(batch, epoch=epoch, batch_index=batch_index))
            if not task_batches:
                raise ValueError("callback.expand_task_batches returned no tasks")
            return task_batches
        return [batch]

    def _compute_output_alignment_loss(
        self,
        student_result: Any,
        teacher_result: Any,
        batch: Any,
    ) -> torch.Tensor:
        custom_loss = getattr(self.callback, "compute_output_alignment_loss", None)
        if callable(custom_loss):
            return custom_loss(student_result, teacher_result, batch)
        aligned = self.callback.extract_alignment_view(student_result, batch)
        teacher_alignment = self.callback.extract_alignment_view(teacher_result, batch)
        return torch.mean((aligned - teacher_alignment.detach()) ** 2)

    def run_epoch(self, loader: Any, *, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        if self.mask_module is not None:
            self.mask_module.train(train)
        total_loss = 0.0
        total_supervised = 0.0
        total_output = 0.0
        total_repr = 0.0
        total_count = 0
        metrics_accumulator: dict[str, float] = {}
        all_subnets = self._enumerate_subnets(epoch)

        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = self.callback.prepare_batch(raw_batch, self.device)
            sampled_subnets = self._sample_subnets(all_subnets, epoch=epoch, batch_index=batch_index)
            task_batches = self._expand_task_batches(batch, epoch=epoch, batch_index=batch_index)
            task_objectives: list[torch.Tensor] = []
            task_loss_rows: list[dict[str, float]] = []
            task_metric_rows: list[dict[str, float]] = []

            for task_batch in task_batches:
                need_repr = bool(self.config.enable_repr_alignment)
                full_result = self._forward(task_batch, width=1.0, depth=1.0, return_encoder_state=need_repr)
                full_supervised = self.callback.compute_supervised_loss(full_result, task_batch)
                teacher_result = self._teacher_forward(task_batch, return_encoder_state=need_repr)
                teacher_representation = (
                    self.callback.extract_representation(teacher_result, task_batch) if need_repr else None
                )
                subnet_terms: list[torch.Tensor] = []
                subnet_supervised_losses: list[torch.Tensor] = []
                subnet_output_losses: list[torch.Tensor] = []
                subnet_repr_losses: list[torch.Tensor] = []

                for subnet in sampled_subnets:
                    if subnet.width_multiplier == 1.0 and subnet.depth_multiplier == 1.0:
                        continue
                    subnet_result = self._forward(
                        task_batch,
                        width=subnet.width_multiplier,
                        depth=subnet.depth_multiplier,
                        return_encoder_state=need_repr,
                    )
                    supervised_loss = self.callback.compute_supervised_loss(subnet_result, task_batch)
                    subnet_supervised_losses.append(supervised_loss.detach())
                    subnet_objective = self.config.supervised_weight * supervised_loss
                    if self.config.enable_output_alignment:
                        output_loss = self._compute_output_alignment_loss(subnet_result, teacher_result, task_batch)
                        subnet_output_losses.append(output_loss.detach())
                        subnet_objective = subnet_objective + self.config.lambda_output * output_loss
                    if self.config.enable_repr_alignment:
                        if self.mask_module is None or teacher_representation is None:
                            raise RuntimeError("repr alignment was enabled without a structural mask module")
                        student_repr = self.callback.extract_representation(subnet_result, task_batch)
                        repr_outputs = self.mask_module(
                            student_repr,
                            teacher_representation,
                            subnet_result.structure_mask,
                        )
                        repr_loss = repr_outputs["loss"]
                        subnet_repr_losses.append(repr_loss.detach())
                        subnet_objective = subnet_objective + self.config.lambda_repr * repr_loss
                    subnet_terms.append(subnet_objective)

                if self.objective_mode == "full_plus_mean_subnets":
                    task_total = full_supervised
                    if subnet_terms:
                        task_total = task_total + torch.stack(subnet_terms).mean()
                elif self.objective_mode == "mean_all":
                    task_total = torch.stack([full_supervised, *subnet_terms]).mean()
                else:
                    raise ValueError(f"unsupported objective_mode '{self.objective_mode}'")

                task_objectives.append(task_total)
                task_loss_rows.append(
                    {
                        "loss": float(task_total.detach().item()),
                        "supervised_loss": (
                            float(torch.stack(subnet_supervised_losses).mean().item())
                            if subnet_supervised_losses
                            else 0.0
                        ),
                        "output_alignment_loss": (
                            float(torch.stack(subnet_output_losses).mean().item())
                            if subnet_output_losses
                            else 0.0
                        ),
                        "repr_alignment_loss": (
                            float(torch.stack(subnet_repr_losses).mean().item())
                            if subnet_repr_losses
                            else 0.0
                        ),
                    }
                )
                task_metric_rows.append(self.callback.compute_metrics(full_result, task_batch))

            total = torch.stack(task_objectives).mean()

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.mask_module is not None:
                    torch.nn.utils.clip_grad_norm_(self.mask_module.parameters(), 1.0)
                self.optimizer.step()
                self._update_ema()

            batch_size = self.callback.batch_size(batch)
            averaged_task_losses = {
                key: sum(row[key] for row in task_loss_rows) / max(len(task_loss_rows), 1)
                for key in ("loss", "supervised_loss", "output_alignment_loss", "repr_alignment_loss")
            }
            averaged_task_metrics = {
                key: sum(metric_row.get(key, 0.0) for metric_row in task_metric_rows) / max(len(task_metric_rows), 1)
                for key in {key for metric_row in task_metric_rows for key in metric_row.keys()}
            }

            total_count += batch_size
            total_loss += averaged_task_losses["loss"] * batch_size
            total_supervised += averaged_task_losses["supervised_loss"] * batch_size
            total_output += averaged_task_losses["output_alignment_loss"] * batch_size
            total_repr += averaged_task_losses["repr_alignment_loss"] * batch_size
            for key, value in averaged_task_metrics.items():
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + float(value) * batch_size

            if train and self.log_every_batches > 0 and (
                batch_index % self.log_every_batches == 0 or batch_index == len(loader)
            ):
                payload = {
                    "event": "train_progress",
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches_per_rank": len(loader),
                    "loss": averaged_task_losses["loss"],
                    "avg_loss": total_loss / max(1, total_count),
                    "sampled_subnets": float(len(sampled_subnets)),
                }
                if "nmse" in averaged_task_metrics:
                    payload["full_nmse"] = float(averaged_task_metrics["nmse"])
                if "nmse" in metrics_accumulator:
                    payload["avg_full_nmse"] = metrics_accumulator["nmse"] / max(1, total_count)
                print(json.dumps(payload, ensure_ascii=True), flush=True)

        metrics = {key: value / max(1, total_count) for key, value in metrics_accumulator.items()}
        metrics.update(
            {
                "loss": total_loss / max(1, total_count),
                "supervised_loss": total_supervised / max(1, total_count),
                "output_alignment_loss": total_output / max(1, total_count),
                "repr_alignment_loss": total_repr / max(1, total_count),
            }
        )
        return metrics

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
            record = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}}
            should_validate = (
                val_loader is not None
                and (epoch % max(1, self.validate_every) == 0 or epoch == int(epochs))
            )
            if should_validate:
                with torch.inference_mode():
                    val_metrics = self.run_epoch(val_loader, epoch=epoch, train=False)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)
        return history
