"""Unified alignment-oriented trainer for isolated elastic models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from elastic_method.core.structures import ElasticizationSpec
from elastic_method.core.subnet import dedupe_subnets, resolve_multiplier_list
from elastic_method.method.mask import StructuralMaskModule
from elastic_method.tasks.protocol import TaskCallback


@dataclass
class MethodConfig:
    supervised_weight: float = 1.0
    lambda_output: float = 1.0
    lambda_repr: float = 0.05
    enable_output_alignment: bool = True
    enable_repr_alignment: bool = True
    use_ema_full_view: bool = True
    ema_momentum: float = 0.996
    initialize_from_full_view: bool = False
    full_view_checkpoint: str | None = None


class AlignmentTrainer:
    """A shared trainer that keeps task-specific details in a callback object."""

    def __init__(
        self,
        model: nn.Module,
        callback: TaskCallback,
        *,
        spec: ElasticizationSpec,
        config: MethodConfig | None = None,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        mask_module: StructuralMaskModule | None = None,
    ) -> None:
        self.model = model
        self.callback = callback
        self.spec = spec
        self.config = config or MethodConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.requires_grad_(True)
        self.mask_module = mask_module
        if self.mask_module is not None:
            self.mask_module.to(self.device)
        if self.config.initialize_from_full_view and self.config.full_view_checkpoint:
            state_dict = torch.load(Path(self.config.full_view_checkpoint), map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
        if self.mask_module is None and self.config.enable_repr_alignment:
            representation_dim = getattr(callback, "representation_dim", None)
            if representation_dim is None:
                raise ValueError("callback.representation_dim is required when repr alignment is enabled")
            max_layers = getattr(self.model, "metadata").total_layers
            self.mask_module = StructuralMaskModule(representation_dim, max_layers=max_layers).to(self.device)
        optimize_params = list(self.model.parameters())
        if self.mask_module is not None:
            optimize_params.extend(self.mask_module.parameters())
        self.optimizer = optimizer or torch.optim.AdamW(optimize_params, lr=1e-3, weight_decay=1e-4)
        self.ema_model = None
        if self.config.use_ema_full_view:
            self.ema_model = deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)

    def _enumerate_subnets(self, epoch: int):
        width_values = resolve_multiplier_list(self.spec.width_multipliers)
        if epoch <= int(self.spec.width_only_epochs):
            depth_values = [1.0]
        else:
            depth_values = resolve_multiplier_list(self.spec.depth_multipliers)
        return dedupe_subnets(width_values, depth_values)

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        online_state = self.model.state_dict()
        for name, ema_value in self.ema_model.state_dict().items():
            value = online_state[name]
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.config.ema_momentum).add_(value.detach(), alpha=1.0 - self.config.ema_momentum)
            else:
                ema_value.copy_(value)

    def _teacher_forward(self, batch: Any, *, return_encoder_state: bool):
        teacher = self.ema_model if self.ema_model is not None else self.model
        with torch.inference_mode():
            return teacher(
                *batch["model_args"],
                width_multiplier=1.0,
                depth_multiplier=1.0,
                return_encoder_state=return_encoder_state,
                **batch["model_kwargs"],
            )

    def _forward(self, batch: Any, *, width: float, depth: float, return_encoder_state: bool):
        return self.model(
            *batch["model_args"],
            width_multiplier=width,
            depth_multiplier=depth,
            return_encoder_state=return_encoder_state,
            **batch["model_kwargs"],
        )

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
        subnets = self._enumerate_subnets(epoch)
        for raw_batch in loader:
            batch = self.callback.prepare_batch(raw_batch, self.device)
            need_repr = bool(self.config.enable_repr_alignment)
            full_result = self._forward(batch, width=1.0, depth=1.0, return_encoder_state=need_repr)
            full_supervised = self.callback.compute_supervised_loss(full_result, batch)
            losses = [full_supervised]
            batch_output = torch.zeros((), device=self.device)
            batch_repr = torch.zeros((), device=self.device)

            teacher_result = self._teacher_forward(batch, return_encoder_state=need_repr)
            teacher_alignment = self.callback.extract_alignment_view(teacher_result, batch)
            teacher_representation = (
                self.callback.extract_representation(teacher_result, batch) if need_repr else None
            )

            for subnet in subnets:
                if subnet.width_multiplier == 1.0 and subnet.depth_multiplier == 1.0:
                    continue
                subnet_result = self._forward(
                    batch,
                    width=subnet.width_multiplier,
                    depth=subnet.depth_multiplier,
                    return_encoder_state=need_repr,
                )
                supervised_loss = self.callback.compute_supervised_loss(subnet_result, batch)
                loss = self.config.supervised_weight * supervised_loss
                total_supervised += float(supervised_loss.detach().item()) * self.callback.batch_size(batch)
                if self.config.enable_output_alignment:
                    aligned = self.callback.extract_alignment_view(subnet_result, batch)
                    output_loss = torch.mean((aligned - teacher_alignment.detach()) ** 2)
                    loss = loss + self.config.lambda_output * output_loss
                    batch_output = batch_output + output_loss.detach()
                if self.config.enable_repr_alignment:
                    if self.mask_module is None or teacher_representation is None:
                        raise RuntimeError("repr alignment was enabled without a structural mask module")
                    student_repr = self.callback.extract_representation(subnet_result, batch)
                    repr_outputs = self.mask_module(
                        student_repr,
                        teacher_representation,
                        subnet_result.structure_mask,
                    )
                    repr_loss = repr_outputs["loss"]
                    loss = loss + self.config.lambda_repr * repr_loss
                    batch_repr = batch_repr + repr_loss.detach()
                losses.append(loss)
            total = torch.stack(losses).mean()

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.mask_module is not None:
                    torch.nn.utils.clip_grad_norm_(self.mask_module.parameters(), 1.0)
                self.optimizer.step()
                self._update_ema()

            batch_size = self.callback.batch_size(batch)
            total_count += batch_size
            total_loss += float(total.detach().item()) * batch_size
            total_output += float(batch_output.item()) * batch_size
            total_repr += float(batch_repr.item()) * batch_size
            metric_result = self.callback.compute_metrics(full_result, batch)
            for key, value in metric_result.items():
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + float(value) * batch_size

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
            if val_loader is not None:
                with torch.inference_mode():
                    val_metrics = self.run_epoch(val_loader, epoch=epoch, train=False)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(record)
        return history
