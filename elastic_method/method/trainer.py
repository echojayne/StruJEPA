"""Trainer for the current two-stage StruJEPA method."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from elastic_method.core.structures import ElasticizationSpec
from elastic_method.core.subnet import dedupe_subnets, resolve_multiplier_list
from elastic_method.method.completion import (
    WidthOperatorCompletionConfig,
    WidthOperatorCompletionModule,
    parse_completion_config,
)
from elastic_method.tasks.protocol import TaskAdapter


@dataclass
class MethodConfig:
    """Configuration for the single maintained StruJEPA recipe."""

    supervised_weight: float = 1.0
    lambda_output: float = 0.0
    lambda_repr: float = 0.0
    validation_seed: int = 0
    subnets_per_batch: int = 0
    adaptive_focus_subnets: int = 0
    hardest_subnet_weight: float = 0.0
    full_view_weight: float = 1.0
    initialize_from_full_view: bool = False
    full_view_checkpoint: str | None = None
    reference_from_resume: bool = False
    subnet_rss_targets_db: dict[str, float] | None = None
    target_margin_db: float = 0.0
    target_gap_loss_weight: float = 0.0
    initial_subnet_loss_multipliers: dict[str, float] | None = None
    initial_priority_subnet_labels: list[str] | None = None
    completion: WidthOperatorCompletionConfig | dict[str, Any] | bool | None = None
    stage_convergence: dict[str, Any] | bool | None = None

    def __post_init__(self) -> None:
        self.completion = parse_completion_config(self.completion)
        if self.subnet_rss_targets_db is None:
            self.subnet_rss_targets_db = {}
        elif not isinstance(self.subnet_rss_targets_db, dict):
            raise TypeError("method.subnet_rss_targets_db must be a dict or None")
        else:
            self.subnet_rss_targets_db = {
                str(label): float(value)
                for label, value in self.subnet_rss_targets_db.items()
            }
        if self.initial_subnet_loss_multipliers is None:
            self.initial_subnet_loss_multipliers = {}
        elif not isinstance(self.initial_subnet_loss_multipliers, dict):
            raise TypeError("method.initial_subnet_loss_multipliers must be a dict or None")
        else:
            self.initial_subnet_loss_multipliers = {
                str(label): max(0.0, float(value))
                for label, value in self.initial_subnet_loss_multipliers.items()
            }
        if self.initial_priority_subnet_labels is None:
            self.initial_priority_subnet_labels = []
        elif not isinstance(self.initial_priority_subnet_labels, list):
            raise TypeError("method.initial_priority_subnet_labels must be a list or None")
        else:
            self.initial_priority_subnet_labels = [
                str(label)
                for label in self.initial_priority_subnet_labels
            ]
        if self.stage_convergence is None:
            self.stage_convergence = {}
        elif isinstance(self.stage_convergence, bool):
            self.stage_convergence = {"enabled": bool(self.stage_convergence)}
        elif not isinstance(self.stage_convergence, dict):
            raise TypeError("method.stage_convergence must be a dict, bool, or None")


class StruJEPATrainer:
    """Task-independent two-stage trainer using a task adapter."""

    def __init__(
        self,
        model: nn.Module,
        task_adapter: TaskAdapter,
        *,
        spec: ElasticizationSpec,
        config: MethodConfig | None = None,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.task_adapter = task_adapter
        self.spec = spec
        self.config = config or MethodConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.requires_grad_(True)
        self._priority_subnet_labels = list(self.config.initial_priority_subnet_labels or [])
        self._target_loss_multipliers = dict(self.config.initial_subnet_loss_multipliers or {})

        if self.config.initialize_from_full_view and self.config.full_view_checkpoint:
            payload = torch.load(Path(self.config.full_view_checkpoint), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "elastic_state_dict" in payload:
                self.model.load_state_dict(payload["elastic_state_dict"], strict=True)
            else:
                state_dict = payload
                if isinstance(payload, dict) and "model_state_dict" in payload:
                    state_dict = payload["model_state_dict"]
                elif isinstance(payload, dict) and "state_dict" in payload:
                    state_dict = payload["state_dict"]
                target = getattr(self.model, "model", self.model)
                target.load_state_dict(state_dict, strict=True)

        self.completion_config = self.config.completion
        self.completion_module: WidthOperatorCompletionModule | None = None
        self.reference_model: nn.Module | None = None
        needs_reference_model = (
            self.completion_config.enabled
            or float(self.config.lambda_output) > 0.0
            or float(self.config.lambda_repr) > 0.0
        )
        if self.completion_config.enabled:
            self.completion_module = WidthOperatorCompletionModule.from_elastic_model(
                self.model,
                self.completion_config,
            ).to(self.device)
        if needs_reference_model:
            self.reference_model = deepcopy(self.model).to(self.device)
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)

        optimize_params = list(self.model.parameters())
        if self.completion_module is not None:
            optimize_params.extend(self.completion_module.parameters())
        self.optimizer = optimizer or torch.optim.AdamW(optimize_params, lr=1e-3, weight_decay=1e-4)
        self._ensure_optimizer_covers_trainable_modules()

    def refresh_reference_from_student(self) -> None:
        self.reference_model = deepcopy(self.model).to(self.device)
        self.reference_model.eval()
        self.reference_model.requires_grad_(False)

    def _ensure_optimizer_covers_trainable_modules(self) -> None:
        """External optimizers must also see internally-created modules."""

        existing = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group.get("params", [])
        }
        missing = [
            parameter
            for module in (self.model, self.completion_module)
            if module is not None
            for parameter in module.parameters()
            if id(parameter) not in existing
        ]
        if not missing:
            return
        defaults = {
            key: value
            for key, value in self.optimizer.param_groups[0].items()
            if key != "params"
        }
        self.optimizer.add_param_group({"params": missing, **defaults})

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "completion_state_dict": (
                self.completion_module.state_dict()
                if self.completion_module is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        model_state = state.get("model_state_dict")
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=strict)
        completion_state = state.get("completion_state_dict")
        if completion_state is not None:
            if self.completion_module is None:
                raise RuntimeError("checkpoint contains completion_state_dict but completion is disabled")
            self.completion_module.load_state_dict(completion_state, strict=strict)

    def _capture_trainable_state(self) -> dict[str, Any]:
        def clone_state(module: nn.Module | None) -> dict[str, torch.Tensor] | None:
            if module is None:
                return None
            return {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }

        return {
            "model_state_dict": clone_state(self.model),
            "completion_state_dict": clone_state(self.completion_module),
        }

    def _restore_trainable_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        model_state = state.get("model_state_dict")
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=True)
        completion_state = state.get("completion_state_dict")
        if completion_state is not None and self.completion_module is not None:
            self.completion_module.load_state_dict(completion_state, strict=True)
        self.model.to(self.device)
        if self.completion_module is not None:
            self.completion_module.to(self.device)

    def _enumerate_subnets(self, epoch: int) -> list[Any]:
        width_values = resolve_multiplier_list(self.spec.width_multipliers)
        depth_values = (
            [1.0]
            if epoch <= int(self.spec.width_only_epochs)
            else resolve_multiplier_list(self.spec.depth_multipliers)
        )
        return dedupe_subnets(width_values, depth_values)

    def _select_subnets_for_batch(
        self,
        all_subnets: list[Any],
        *,
        epoch: int,
        batch_index: int,
        train: bool,
    ) -> list[Any]:
        limit = int(self.config.subnets_per_batch)
        if not train or limit <= 0 or limit >= len(all_subnets):
            return list(all_subnets)

        limit = max(1, limit)
        full = next(
            (
                subnet
                for subnet in all_subnets
                if float(subnet.width_multiplier) == 1.0
                and float(subnet.depth_multiplier) == 1.0
            ),
            all_subnets[0],
        )
        selected = [full]
        if limit == 1:
            return selected

        non_full = [subnet for subnet in all_subnets if subnet is not full]
        smallest = min(
            non_full,
            key=lambda subnet: (
                float(subnet.width_multiplier) * float(subnet.depth_multiplier),
                float(subnet.width_multiplier),
                float(subnet.depth_multiplier),
            ),
        )
        selected.append(smallest)
        focus_count = max(0, int(self.config.adaptive_focus_subnets))
        priority_labels = self._priority_subnet_labels[:focus_count]
        for label in priority_labels:
            subnet = next(
                (
                    candidate
                    for candidate in non_full
                    if self._subnet_label(
                        candidate.width_multiplier,
                        candidate.depth_multiplier,
                    )
                    == label
                ),
                None,
            )
            if subnet is not None and subnet not in selected and len(selected) < limit:
                selected.append(subnet)
        if len(selected) >= limit:
            return selected

        candidates = [subnet for subnet in non_full if subnet is not smallest]
        candidates = [subnet for subnet in candidates if subnet not in selected]
        remaining = min(limit - len(selected), len(candidates))
        if remaining <= 0:
            return selected

        offset = (
            (max(1, int(epoch)) - 1) * remaining
            + (max(1, int(batch_index)) - 1) * remaining
        ) % len(candidates)
        selected.extend(
            candidates[(offset + index) % len(candidates)]
            for index in range(remaining)
        )
        return selected

    def _update_adaptive_subnet_focus(self, record: dict[str, Any]) -> None:
        focus_count = max(0, int(self.config.adaptive_focus_subnets))
        prefix = "val_subnet_"
        suffix = "_rss_loss_db"
        rows: list[tuple[str, float]] = []
        target_gaps: dict[str, float] = {}
        for key, value in record.items():
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            label = key[len(prefix) : -len(suffix)]
            rss_loss_db = float(value)
            target_gap = self._target_gap_db(label, rss_loss_db)
            if target_gap is not None:
                target_gaps[label] = target_gap
            rows.append((label, target_gap if target_gap is not None else rss_loss_db))

        loss_weight = max(0.0, float(self.config.target_gap_loss_weight))
        max_positive_gap = max((max(0.0, gap) for gap in target_gaps.values()), default=0.0)
        if loss_weight > 0.0 and max_positive_gap > 0.0:
            self._target_loss_multipliers = {
                label: 1.0 + loss_weight * max(0.0, gap) / max_positive_gap
                for label, gap in target_gaps.items()
            }
        else:
            self._target_loss_multipliers = {
                label: 1.0
                for label in target_gaps
            }
        if self._target_loss_multipliers:
            record["target_loss_multipliers"] = dict(self._target_loss_multipliers)

        if focus_count <= 0:
            self._priority_subnet_labels = []
            return
        rows.sort(key=lambda item: (-item[1], item[0]))
        self._priority_subnet_labels = [
            label
            for label, _value in rows
            if label != self._subnet_label(1.0, 1.0)
        ][:focus_count]
        record["adaptive_focus_subnet_labels"] = list(self._priority_subnet_labels)

    def _expand_task_batches(self, batch: Any, *, epoch: int, batch_index: int) -> list[Any]:
        return [batch]

    def _combine_task_losses(
        self,
        full_loss: torch.Tensor,
        subnet_losses: list[torch.Tensor],
        *,
        full_label: str | None = None,
        subnet_labels: list[str] | None = None,
    ) -> torch.Tensor:
        if subnet_labels is not None and len(subnet_labels) != len(subnet_losses):
            raise ValueError("subnet_labels must match subnet_losses")
        full_target_weight = self._target_loss_multipliers.get(str(full_label), 1.0)
        subnet_target_weights = [
            self._target_loss_multipliers.get(str(label), 1.0)
            for label in (subnet_labels or [None] * len(subnet_losses))
        ]
        full_weight = max(0.0, float(self.config.full_view_weight)) * full_target_weight
        denominator = full_weight + sum(subnet_target_weights)
        if denominator <= 0.0:
            mean_loss = full_loss
        else:
            mean_loss = (
                full_weight * full_loss
                + sum(
                    (
                        weight * subnet_loss
                        for weight, subnet_loss in zip(subnet_target_weights, subnet_losses)
                    ),
                    start=torch.zeros_like(full_loss),
                )
            ) / denominator

        hardest_weight = min(1.0, max(0.0, float(self.config.hardest_subnet_weight)))
        if hardest_weight <= 0.0:
            return mean_loss
        hardest_loss = torch.stack(
            [
                full_target_weight * full_loss,
                *(
                    weight * subnet_loss
                    for weight, subnet_loss in zip(subnet_target_weights, subnet_losses)
                ),
            ]
        ).max()
        return (1.0 - hardest_weight) * mean_loss + hardest_weight * hardest_loss

    def _forward(
        self,
        batch: Any,
        *,
        width: float,
        depth: float,
        completion_module: nn.Module | None = None,
        trace_blocks: bool = False,
    ):
        return self.model(
            *batch["model_args"],
            width_multiplier=width,
            depth_multiplier=depth,
            return_encoder_state=False,
            completion_module=completion_module,
            trace_blocks=trace_blocks,
            **batch["model_kwargs"],
        )

    def _reference_forward(self, batch: Any, *, trace_blocks: bool):
        if self.reference_model is None:
            raise RuntimeError("completion warmup requires a frozen reference model")
        with torch.inference_mode():
            return self.reference_model(
                *batch["model_args"],
                width_multiplier=1.0,
                depth_multiplier=1.0,
                return_encoder_state=False,
                trace_blocks=trace_blocks,
                **batch["model_kwargs"],
            )

    def _stage(self, epoch: int) -> str:
        warmup_epochs = max(0, int(self.completion_config.stage_epochs.warmup_completion))
        return "completion_warmup" if epoch <= warmup_epochs else "subnet_training"

    def _stage_subnets(self, *, stage: str, epoch: int) -> list[Any]:
        widths = resolve_multiplier_list(self.spec.width_multipliers)
        depths = (
            [1.0]
            if stage in {"completion_warmup", "output_warmup"}
            else resolve_multiplier_list(self.spec.depth_multipliers)
        )
        return dedupe_subnets(widths, depths)

    def _combine_subnet_losses(self, subnet_losses: list[torch.Tensor]) -> torch.Tensor:
        if not subnet_losses:
            return torch.zeros((), device=self.device)
        mean_loss = torch.stack(subnet_losses).mean()
        hardest_weight = min(1.0, max(0.0, float(self.config.hardest_subnet_weight)))
        if hardest_weight <= 0.0:
            return mean_loss
        hardest_loss = torch.stack(subnet_losses).max()
        return (1.0 - hardest_weight) * mean_loss + hardest_weight * hardest_loss

    def _configure_trainability(self, *, stage: str, train: bool) -> None:
        self.model.requires_grad_(train)
        self.model.train(train)
        if self.completion_module is not None:
            train_completion = train and stage == "completion_warmup"
            self.completion_module.requires_grad_(train_completion)
            self.completion_module.train(train_completion)
        if self.reference_model is not None:
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)

    def _trace_residual_loss(self, result: Any, reference_result: Any, key: str) -> torch.Tensor:
        reference_traces: dict[int, list[torch.Tensor]] = {}
        for trace in reference_result.aux.get("block_traces", []):
            if key in trace:
                reference_traces.setdefault(int(trace["layer_index"]), []).append(trace[key].detach())

        losses: list[torch.Tensor] = []
        for trace in result.aux.get("block_traces", []):
            if key not in trace:
                continue
            candidates = reference_traces.get(int(trace["layer_index"]), [])
            match_index = next(
                (index for index, candidate in enumerate(candidates) if candidate.shape == trace[key].shape),
                -1,
            )
            if match_index >= 0:
                losses.append(torch.mean((trace[key] - candidates.pop(match_index)) ** 2))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=self.device)

    def _output_alignment_loss(
        self,
        result: Any,
        reference_result: Any,
        batch: Any,
    ) -> torch.Tensor:
        custom = getattr(self.task_adapter, "compute_output_alignment_loss", None)
        if callable(custom):
            return custom(result, reference_result, batch)
        extract = getattr(self.task_adapter, "extract_alignment_view", None)
        if callable(extract):
            student = extract(result, batch)
            teacher = extract(reference_result, batch)
        else:
            student = result.model_output
            teacher = reference_result.model_output
        return torch.mean((student - teacher.detach()) ** 2)

    def _completion_weight_loss(self, result: Any) -> torch.Tensor:
        losses = list(result.aux.get("completion_losses", []))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=self.device)

    @staticmethod
    def _subnet_label(width: float, depth: float) -> str:
        return f"w{float(width):g}_d{float(depth):g}"

    def _target_gap_db(self, label: str, rss_loss_db: float) -> float | None:
        targets = self.config.subnet_rss_targets_db or {}
        if label not in targets:
            return None
        return float(rss_loss_db) - float(targets[label]) + float(self.config.target_margin_db)

    def _clip_trainable_gradients(self) -> None:
        parameters = [
            parameter
            for module in (self.model, self.completion_module)
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)

    def _optimizer_step(self, loss: torch.Tensor, *, train: bool) -> None:
        if not train or not loss.requires_grad:
            return
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_trainable_gradients()
        self.optimizer.step()

    def _run_completion_warmup(
        self,
        loader: Any,
        *,
        epoch: int,
        train: bool,
    ) -> dict[str, float]:
        if self.completion_module is None:
            raise RuntimeError("completion warmup requires a completion module")

        total_loss = 0.0
        total_weight = 0.0
        total_attn = 0.0
        total_ffn = 0.0
        total_supervised = 0.0
        total_output_alignment = 0.0
        total_count = 0
        metrics_accumulator: dict[str, float] = {}
        all_subnets = self._stage_subnets(stage="completion_warmup", epoch=epoch)

        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = self.task_adapter.prepare_batch(raw_batch, self.device)
            sampled_subnets = self._select_subnets_for_batch(
                all_subnets,
                epoch=epoch,
                batch_index=batch_index,
                train=train,
            )
            task_batches = self._expand_task_batches(batch, epoch=epoch, batch_index=batch_index)
            task_losses: list[torch.Tensor] = []
            task_weight: list[float] = []
            task_attn: list[float] = []
            task_ffn: list[float] = []
            task_supervised: list[float] = []
            task_output_alignment: list[float] = []
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for task_batch in task_batches:
                    reference_result = self._reference_forward(task_batch, trace_blocks=True)
                    subnet_losses: list[torch.Tensor] = []
                    weight_losses: list[torch.Tensor] = []
                    attn_losses: list[torch.Tensor] = []
                    ffn_losses: list[torch.Tensor] = []
                    for subnet in sampled_subnets:
                        if subnet.width_multiplier == 1.0:
                            continue
                        completed_result = self._forward(
                            task_batch,
                            width=subnet.width_multiplier,
                            depth=1.0,
                            completion_module=self.completion_module,
                            trace_blocks=True,
                        )
                        weight_loss = self._completion_weight_loss(completed_result)
                        attn_loss = self._trace_residual_loss(
                            completed_result,
                            reference_result,
                            "attention_residual",
                        )
                        ffn_loss = self._trace_residual_loss(
                            completed_result,
                            reference_result,
                            "ffn_residual",
                        )
                        subnet_losses.append(
                            self.completion_config.lambda_weight * weight_loss
                            + self.completion_config.lambda_attn_residual * attn_loss
                            + self.completion_config.lambda_ffn_residual * ffn_loss
                        )
                        weight_losses.append(weight_loss.detach())
                        attn_losses.append(attn_loss.detach())
                        ffn_losses.append(ffn_loss.detach())

                    task_loss = self._combine_subnet_losses(subnet_losses)
                    task_losses.append(task_loss)
                    task_weight.append(float(torch.stack(weight_losses).mean().item()) if weight_losses else 0.0)
                    task_attn.append(float(torch.stack(attn_losses).mean().item()) if attn_losses else 0.0)
                    task_ffn.append(float(torch.stack(ffn_losses).mean().item()) if ffn_losses else 0.0)
                    task_supervised.append(0.0)
                    task_output_alignment.append(0.0)

                loss = torch.stack(task_losses).mean()

            self._optimizer_step(loss, train=train)
            batch_size = self.task_adapter.batch_size(batch)
            loss_value = float(loss.detach().item())
            weight_value = sum(task_weight) / max(1, len(task_weight))
            attn_value = sum(task_attn) / max(1, len(task_attn))
            ffn_value = sum(task_ffn) / max(1, len(task_ffn))
            supervised_value = sum(task_supervised) / max(1, len(task_supervised))
            output_alignment_value = sum(task_output_alignment) / max(1, len(task_output_alignment))
            total_count += batch_size
            total_loss += loss_value * batch_size
            total_weight += weight_value * batch_size
            total_attn += attn_value * batch_size
            total_ffn += ffn_value * batch_size
            total_supervised += supervised_value * batch_size
            total_output_alignment += output_alignment_value * batch_size
            self._log_progress(
                stage="completion_warmup",
                epoch=epoch,
                batch_index=batch_index,
                loader=loader,
                loss=loss_value,
                sampled_subnets=len(sampled_subnets),
            )

        metrics = {key: value / max(1, total_count) for key, value in metrics_accumulator.items()}
        metrics.update(
            {
                "loss": total_loss / max(1, total_count),
                "weight_completion_loss": total_weight / max(1, total_count),
                "attn_residual_loss": total_attn / max(1, total_count),
                "ffn_residual_loss": total_ffn / max(1, total_count),
                "supervised_loss": total_supervised / max(1, total_count),
                "output_alignment_loss": total_output_alignment / max(1, total_count),
            }
        )
        return metrics

    def _run_output_warmup(
        self,
        loader: Any,
        *,
        epoch: int,
        train: bool,
    ) -> dict[str, float]:
        total_loss = 0.0
        total_output_alignment = 0.0
        total_count = 0
        metrics_accumulator: dict[str, float] = {}
        all_subnets = self._stage_subnets(stage="output_warmup", epoch=epoch)

        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = self.task_adapter.prepare_batch(raw_batch, self.device)
            sampled_subnets = self._select_subnets_for_batch(
                all_subnets,
                epoch=epoch,
                batch_index=batch_index,
                train=train,
            )
            task_batches = self._expand_task_batches(batch, epoch=epoch, batch_index=batch_index)
            objectives: list[torch.Tensor] = []
            output_alignment_values: list[float] = []
            metric_rows: list[dict[str, float]] = []

            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for task_batch in task_batches:
                    reference_result = self._reference_forward(task_batch, trace_blocks=False)
                    full_result = self._forward(
                        task_batch,
                        width=1.0,
                        depth=1.0,
                        trace_blocks=False,
                    )
                    full_output_alignment = self._output_alignment_loss(
                        full_result,
                        reference_result,
                        task_batch,
                    )
                    subnet_losses: list[torch.Tensor] = [
                        max(0.0, float(self.config.full_view_weight))
                        * float(self.config.lambda_output)
                        * full_output_alignment
                    ]
                    output_alignments = [full_output_alignment]
                    metric_rows.append(self.task_adapter.compute_metrics(full_result, task_batch))

                    for subnet in sampled_subnets:
                        if subnet.width_multiplier == 1.0 and subnet.depth_multiplier == 1.0:
                            continue
                        subnet_result = self._forward(
                            task_batch,
                            width=subnet.width_multiplier,
                            depth=subnet.depth_multiplier,
                            trace_blocks=False,
                        )
                        output_alignment = self._output_alignment_loss(
                            subnet_result,
                            reference_result,
                            task_batch,
                        )
                        subnet_losses.append(float(self.config.lambda_output) * output_alignment)
                        output_alignments.append(output_alignment)
                        metric_rows.append(self.task_adapter.compute_metrics(subnet_result, task_batch))

                    objectives.append(self._combine_subnet_losses(subnet_losses))
                    output_alignment_values.append(
                        float(torch.stack(output_alignments).mean().detach().item())
                    )
                loss = torch.stack(objectives).mean()

            self._optimizer_step(loss, train=train)
            batch_size = self.task_adapter.batch_size(batch)
            loss_value = float(loss.detach().item())
            output_alignment_value = (
                sum(output_alignment_values) / max(1, len(output_alignment_values))
            )
            total_count += batch_size
            total_loss += loss_value * batch_size
            total_output_alignment += output_alignment_value * batch_size
            for key in {key for row in metric_rows for key in row}:
                value = sum(row.get(key, 0.0) for row in metric_rows) / max(1, len(metric_rows))
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + value * batch_size
            self._log_progress(
                stage="output_warmup",
                epoch=epoch,
                batch_index=batch_index,
                loader=loader,
                loss=loss_value,
                sampled_subnets=len(sampled_subnets),
            )

        metrics = {key: value / max(1, total_count) for key, value in metrics_accumulator.items()}
        metrics.update(
            {
                "loss": total_loss / max(1, total_count),
                "output_alignment_loss": total_output_alignment / max(1, total_count),
                "supervised_loss": 0.0,
            }
        )
        return metrics

    def _run_subnet_training(
        self,
        loader: Any,
        *,
        epoch: int,
        train: bool,
    ) -> dict[str, float]:
        total_loss = 0.0
        total_task_loss = 0.0
        total_output_alignment = 0.0
        total_repr_alignment = 0.0
        total_count = 0
        subnet_metric_sums: dict[str, dict[str, float]] = {}
        subnet_metric_counts: dict[str, int] = {}
        all_subnets = self._stage_subnets(stage="subnet_training", epoch=epoch)
        trace_blocks = float(self.config.lambda_repr) > 0.0
        needs_reference = (
            float(self.config.lambda_output) > 0.0
            or float(self.config.lambda_repr) > 0.0
        )

        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = self.task_adapter.prepare_batch(raw_batch, self.device)
            sampled_subnets = self._select_subnets_for_batch(
                all_subnets,
                epoch=epoch,
                batch_index=batch_index,
                train=train,
            )
            task_batches = self._expand_task_batches(batch, epoch=epoch, batch_index=batch_index)
            objectives: list[torch.Tensor] = []
            task_loss_values: list[float] = []
            output_alignment_values: list[float] = []
            repr_alignment_values: list[float] = []
            batch_metric_rows: dict[str, list[dict[str, float]]] = {}

            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for task_batch in task_batches:
                    reference_result = (
                        self._reference_forward(task_batch, trace_blocks=trace_blocks)
                        if needs_reference
                        else None
                    )
                    full_result = self._forward(
                        task_batch,
                        width=1.0,
                        depth=1.0,
                        trace_blocks=trace_blocks,
                    )
                    full_loss = self.task_adapter.compute_supervised_loss(full_result, task_batch)
                    full_output_alignment = (
                        self._output_alignment_loss(
                            full_result,
                            reference_result,
                            task_batch,
                        )
                        if reference_result is not None
                        else torch.zeros((), device=self.device)
                    )
                    full_repr_alignment = (
                        (
                            self._trace_residual_loss(full_result, reference_result, "attention_residual")
                            + self._trace_residual_loss(full_result, reference_result, "ffn_residual")
                        )
                        if reference_result is not None and trace_blocks
                        else torch.zeros((), device=self.device)
                    )
                    full_objective = (
                        full_loss
                        + float(self.config.lambda_output) * full_output_alignment
                        + float(self.config.lambda_repr) * full_repr_alignment
                    )
                    subnet_losses: list[torch.Tensor] = []
                    subnet_labels: list[str] = []
                    output_alignments = [full_output_alignment]
                    repr_alignments = [full_repr_alignment]
                    full_label = self._subnet_label(1.0, 1.0)
                    batch_metric_rows.setdefault(full_label, []).append(
                        self.task_adapter.compute_metrics(full_result, task_batch)
                    )
                    for subnet in sampled_subnets:
                        if subnet.width_multiplier == 1.0 and subnet.depth_multiplier == 1.0:
                            continue
                        subnet_result = self._forward(
                            task_batch,
                            width=subnet.width_multiplier,
                            depth=subnet.depth_multiplier,
                            trace_blocks=trace_blocks,
                        )
                        supervised_loss = self.task_adapter.compute_supervised_loss(subnet_result, task_batch)
                        output_alignment = (
                            self._output_alignment_loss(
                                subnet_result,
                                reference_result,
                                task_batch,
                            )
                            if reference_result is not None
                            else torch.zeros((), device=self.device)
                        )
                        repr_alignment = (
                            (
                                self._trace_residual_loss(
                                    subnet_result,
                                    reference_result,
                                    "attention_residual",
                                )
                                + self._trace_residual_loss(
                                    subnet_result,
                                    reference_result,
                                    "ffn_residual",
                                )
                            )
                            if reference_result is not None and trace_blocks
                            else torch.zeros((), device=self.device)
                        )
                        subnet_losses.append(
                            float(self.config.supervised_weight) * supervised_loss
                            + float(self.config.lambda_output) * output_alignment
                            + float(self.config.lambda_repr) * repr_alignment
                        )
                        output_alignments.append(output_alignment)
                        repr_alignments.append(repr_alignment)
                        subnet_label = self._subnet_label(
                            subnet.width_multiplier,
                            subnet.depth_multiplier,
                        )
                        subnet_labels.append(subnet_label)
                        batch_metric_rows.setdefault(subnet_label, []).append(
                            self.task_adapter.compute_metrics(subnet_result, task_batch)
                        )
                    objective = self._combine_task_losses(
                        full_objective,
                        subnet_losses,
                        full_label=full_label,
                        subnet_labels=subnet_labels,
                    )
                    objectives.append(objective)
                    task_loss_values.append(float(objective.detach().item()))
                    output_alignment_values.append(float(torch.stack(output_alignments).mean().detach().item()))
                    repr_alignment_values.append(float(torch.stack(repr_alignments).mean().detach().item()))
                loss = torch.stack(objectives).mean()

            self._optimizer_step(loss, train=train)
            batch_size = self.task_adapter.batch_size(batch)
            loss_value = float(loss.detach().item())
            task_loss_value = sum(task_loss_values) / max(1, len(task_loss_values))
            output_alignment_value = sum(output_alignment_values) / max(1, len(output_alignment_values))
            repr_alignment_value = sum(repr_alignment_values) / max(1, len(repr_alignment_values))
            total_count += batch_size
            total_loss += loss_value * batch_size
            total_task_loss += task_loss_value * batch_size
            total_output_alignment += output_alignment_value * batch_size
            total_repr_alignment += repr_alignment_value * batch_size
            for label, rows in batch_metric_rows.items():
                sums = subnet_metric_sums.setdefault(label, {})
                subnet_metric_counts[label] = subnet_metric_counts.get(label, 0) + batch_size
                for key in {key for row in rows for key in row}:
                    value = sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))
                    sums[key] = sums.get(key, 0.0) + value * batch_size
            self._log_progress(
                stage="subnet_training",
                epoch=epoch,
                batch_index=batch_index,
                loader=loader,
                loss=loss_value,
                sampled_subnets=len(sampled_subnets),
            )

        subnet_metrics = {
            label: {
                key: value / max(1, subnet_metric_counts.get(label, 0))
                for key, value in sums.items()
            }
            for label, sums in subnet_metric_sums.items()
        }
        full_metrics = subnet_metrics.get(self._subnet_label(1.0, 1.0), {})
        metrics = dict(full_metrics)
        metric_keys = {key for row in subnet_metrics.values() for key in row}
        for key in metric_keys:
            values = [float(row[key]) for row in subnet_metrics.values() if key in row]
            if values:
                metrics[f"elastic_{key}"] = sum(values) / len(values)
        rss_values = [
            float(row["rss_loss_db"])
            for row in subnet_metrics.values()
            if "rss_loss_db" in row
        ]
        if rss_values:
            metrics["worst_rss_loss_db"] = max(rss_values)
        for label, row in subnet_metrics.items():
            for key, value in row.items():
                metrics[f"subnet_{label}_{key}"] = float(value)
        target_gaps = {
            label: gap
            for label, row in subnet_metrics.items()
            if "rss_loss_db" in row
            and (gap := self._target_gap_db(label, float(row["rss_loss_db"]))) is not None
        }
        if target_gaps:
            metrics["worst_target_gap_db"] = max(target_gaps.values())
            metrics["target_subnet_count"] = float(len(target_gaps))
            for label, gap in target_gaps.items():
                metrics[f"subnet_{label}_target_gap_db"] = float(gap)
        metrics.update(
            {
                "loss": total_loss / max(1, total_count),
                "task_loss": total_task_loss / max(1, total_count),
                "output_alignment_loss": total_output_alignment / max(1, total_count),
                "repr_alignment_loss": total_repr_alignment / max(1, total_count),
            }
        )
        return metrics

    def _log_progress(
        self,
        *,
        stage: str,
        epoch: int,
        batch_index: int,
        loader: Any,
        loss: float,
        sampled_subnets: int,
    ) -> None:
        log_every = int(getattr(self, "log_every_batches", 0))
        if log_every <= 0 or (batch_index % log_every != 0 and batch_index != len(loader)):
            return
        print(
            json.dumps(
                {
                    "event": "train_progress",
                    "stage": stage,
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches_per_rank": len(loader),
                    "loss": loss,
                    "sampled_subnets": float(sampled_subnets),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    def run_epoch(self, loader: Any, *, epoch: int, train: bool) -> dict[str, float]:
        stage = self._stage(epoch) if self.completion_config.enabled else "subnet_training"
        return self.run_stage_epoch(loader, stage=stage, epoch=epoch, train=train)

    def run_stage_epoch(
        self,
        loader: Any,
        *,
        stage: str,
        epoch: int,
        train: bool,
    ) -> dict[str, float]:
        self._configure_trainability(stage=stage, train=train)
        if stage == "completion_warmup":
            return self._run_completion_warmup(loader, epoch=epoch, train=train)
        if stage == "output_warmup":
            return self._run_output_warmup(loader, epoch=epoch, train=train)
        return self._run_subnet_training(loader, epoch=epoch, train=train)

    def _run_validation_stage_epoch(
        self,
        loader: Any,
        *,
        stage: str,
        epoch: int,
    ) -> dict[str, float]:
        seed = int(self.config.validation_seed)
        if seed <= 0:
            return self.run_stage_epoch(loader, stage=stage, epoch=epoch, train=False)
        devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return self.run_stage_epoch(loader, stage=stage, epoch=epoch, train=False)

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
            if val_loader is not None:
                stage = self._stage(epoch) if self.completion_config.enabled else "subnet_training"
                val_metrics = self._run_validation_stage_epoch(
                    val_loader,
                    stage=stage,
                    epoch=epoch,
                )
                record.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(record)
        return history

    def _stage_convergence_config(self, stage: str) -> dict[str, Any]:
        raw = dict(self.config.stage_convergence or {})
        stage_raw = dict(raw.get(stage, {}))
        stage_epochs = self.completion_config.stage_epochs
        default_epochs = (
            int(stage_epochs.warmup_completion)
            if stage == "completion_warmup"
            else int(stage_epochs.subnet_training)
        )
        default_epochs = max(1, default_epochs)
        metric = "val_loss" if stage in {"completion_warmup", "output_warmup"} else "val_task_loss"
        return {
            "enabled": bool(raw.get("enabled", False)),
            "metric": str(stage_raw.get("metric", metric)),
            "min_epochs": int(stage_raw.get("min_epochs", default_epochs)),
            "max_epochs": int(stage_raw.get("max_epochs", default_epochs)),
            "patience": int(stage_raw.get("patience", 0)),
            "min_delta": float(stage_raw.get("min_delta", 0.0)),
            "target_metric_value": stage_raw.get("target_metric_value", stage_raw.get("target", None)),
            "restore_best": bool(stage_raw.get("restore_best", raw.get("restore_best", True))),
            "require_converged": bool(stage_raw.get("require_converged", raw.get("require_converged", True))),
        }

    @staticmethod
    def _is_better_metric(value: float, best: float, *, min_delta: float) -> bool:
        return float(value) < float(best) - float(min_delta)

    def _fit_stage_until_converged(
        self,
        train_loader: Any,
        *,
        stage: str,
        start_epoch: int,
        val_loader: Any | None,
        history: list[dict[str, Any]],
        reports: list[dict[str, Any]],
        on_epoch_end: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        cfg = self._stage_convergence_config(stage)
        min_epochs = max(1, int(cfg["min_epochs"]))
        max_epochs = max(min_epochs, int(cfg["max_epochs"]))
        patience = max(0, int(cfg["patience"]))
        metric_key = str(cfg["metric"])
        min_delta = float(cfg["min_delta"])
        target_metric_value = cfg.get("target_metric_value")
        target_metric = float(target_metric_value) if target_metric_value is not None else None

        prior_records = [record for record in history if str(record.get("stage", "")) == stage]
        best_metric = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for record in prior_records:
            try:
                candidate = float(record["convergence_metric_value"])
                candidate_epoch = int(float(record["stage_epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            if candidate < best_metric:
                best_metric = candidate
                best_epoch = candidate_epoch
        if prior_records:
            try:
                stale_epochs = int(float(prior_records[-1].get("convergence_stale_epochs", 0)))
            except (TypeError, ValueError):
                stale_epochs = 0
        best_state: dict[str, Any] | None = (
            self._capture_trainable_state()
            if prior_records
            and int(float(prior_records[-1].get("convergence_best_stage_epoch", -1))) == int(
                float(prior_records[-1].get("stage_epoch", -2))
            )
            else None
        )
        converged = False
        convergence_reason = ""
        epoch = int(start_epoch)

        completed_stage_epochs = max(
            (int(float(record.get("stage_epoch", 0))) for record in prior_records),
            default=0,
        )
        for stage_epoch in range(completed_stage_epochs + 1, max_epochs + 1):
            epoch += 1
            train_metrics = self.run_stage_epoch(
                train_loader,
                stage=stage,
                epoch=epoch,
                train=True,
            )
            record: dict[str, Any] = {
                "epoch": float(epoch),
                "stage": stage,
                "stage_epoch": float(stage_epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
            if val_loader is not None:
                val_metrics = self._run_validation_stage_epoch(
                    val_loader,
                    stage=stage,
                    epoch=epoch,
                )
                record.update({f"val_{key}": value for key, value in val_metrics.items()})
                if stage == "subnet_training":
                    self._update_adaptive_subnet_focus(record)

            if metric_key not in record:
                raise KeyError(
                    f"convergence metric '{metric_key}' missing for stage '{stage}'; "
                    f"available keys={sorted(record)}"
                )
            metric_value = float(record[metric_key])
            improved = self._is_better_metric(metric_value, best_metric, min_delta=min_delta)
            if improved:
                best_metric = metric_value
                best_epoch = stage_epoch
                best_state = self._capture_trainable_state()
                stale_epochs = 0
            else:
                stale_epochs += 1

            record.update(
                {
                    "convergence_metric": metric_key,
                    "convergence_metric_value": metric_value,
                    "convergence_best_metric": best_metric,
                    "convergence_best_stage_epoch": float(best_epoch),
                    "convergence_stale_epochs": float(stale_epochs),
                    "convergence_improved": bool(improved),
                }
            )
            history.append(record)
            if on_epoch_end is not None:
                on_epoch_end(history, reports)

            if stage_epoch >= min_epochs:
                if target_metric is not None and metric_value <= target_metric:
                    converged = True
                    convergence_reason = "target_metric"
                    break
                if stale_epochs >= patience:
                    converged = True
                    convergence_reason = "plateau"
                    break

        if bool(cfg["restore_best"]) and best_state is not None:
            self._restore_trainable_state(best_state)

        report = {
            "stage": stage,
            "metric": metric_key,
            "best_metric": best_metric,
            "best_stage_epoch": best_epoch,
            "epochs_ran": int(history[-1]["stage_epoch"]) if history else 0,
            "converged": bool(converged),
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "target_metric_value": target_metric,
            "convergence_reason": convergence_reason,
            "restored_best": bool(cfg["restore_best"] and best_state is not None),
        }
        if bool(cfg["require_converged"]) and not converged:
            raise RuntimeError(
                f"stage '{stage}' did not converge within {max_epochs} epochs: "
                f"best {metric_key}={best_metric} at stage_epoch={best_epoch}"
            )
        return epoch, report

    def fit_until_converged(
        self,
        train_loader: Any,
        *,
        val_loader: Any | None = None,
        on_epoch_end: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
        initial_history: list[dict[str, Any]] | None = None,
        initial_stage_reports: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not bool((self.config.stage_convergence or {}).get("enabled", False)):
            fixed_epochs = (
                int(self.completion_config.stage_epochs.warmup_completion)
                + int(self.completion_config.stage_epochs.subnet_training)
                if self.completion_config.enabled
                else 1
            )
            return self.fit(train_loader, epochs=max(1, fixed_epochs), val_loader=val_loader), []

        requested_stages = (self.config.stage_convergence or {}).get("stages")
        if requested_stages is None:
            stages = []
            if self.completion_config.enabled and int(self.completion_config.stage_epochs.warmup_completion) != 0:
                stages.append("completion_warmup")
            stages.append("subnet_training")
        else:
            if not isinstance(requested_stages, (list, tuple)) or not requested_stages:
                raise ValueError("method.stage_convergence.stages must be a non-empty list")
            stages = [str(stage) for stage in requested_stages]
            invalid = [
                stage
                for stage in stages
                if stage not in {"completion_warmup", "output_warmup", "subnet_training"}
            ]
            if invalid:
                raise ValueError(f"unsupported convergence stages: {invalid}")
            if "completion_warmup" in stages and not self.completion_config.enabled:
                raise ValueError("completion_warmup requires method.completion.enabled=true")
            if "output_warmup" in stages and float(self.config.lambda_output) <= 0.0:
                raise ValueError("output_warmup requires method.lambda_output > 0")

        history: list[dict[str, Any]] = list(initial_history or [])
        reports: list[dict[str, Any]] = list(initial_stage_reports or [])
        epoch = max((int(float(record.get("epoch", 0))) for record in history), default=0)
        for stage in stages:
            completed_report = next(
                (
                    report
                    for report in reports
                    if str(report.get("stage", "")) == stage and bool(report.get("converged", False))
                ),
                None,
            )
            if completed_report is not None:
                continue
            epoch, report = self._fit_stage_until_converged(
                train_loader,
                stage=stage,
                start_epoch=epoch,
                val_loader=val_loader,
                history=history,
                reports=reports,
                on_epoch_end=on_epoch_end,
            )
            reports.append(report)
        return history, reports
