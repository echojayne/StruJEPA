"""AMP-aware StruJEPA trainer specialized for image classification."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from elastic_method.method.trainer import MethodConfig, StruJEPATrainer
from elastic_method.tasks.protocol import TaskAdapter


@dataclass
class ClassificationTrainerConfig:
    amp_dtype: str = "bf16"
    grad_clip_norm: float = 1.0
    finetune_backbone: bool = True


class ClassificationStruJEPATrainer(StruJEPATrainer):
    """ImageNet trainer with AMP and optional backbone freezing."""

    def __init__(
        self,
        model: torch.nn.Module,
        task_adapter: TaskAdapter,
        *,
        spec,
        method_config: MethodConfig | None = None,
        trainer_config: ClassificationTrainerConfig | None = None,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.trainer_config = trainer_config or ClassificationTrainerConfig()
        super().__init__(
            model,
            task_adapter,
            spec=spec,
            config=method_config,
            device=device,
            optimizer=optimizer,
        )
        self.method_config = self.config
        self.use_fp16 = self.device.type == "cuda" and self.trainer_config.amp_dtype.lower() == "fp16"
        self.use_bf16 = self.device.type == "cuda" and self.trainer_config.amp_dtype.lower() == "bf16"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None
        if not self.trainer_config.finetune_backbone:
            self.model.model.backbone.requires_grad_(False)
            if self.reference_model is not None:
                self.reference_model.model.backbone.requires_grad_(False)

    def _autocast(self):
        if self.use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _forward(self, *args, **kwargs):
        with self._autocast():
            return super()._forward(*args, **kwargs)

    def _reference_forward(self, batch: Any, *, trace_blocks: bool):
        with self._autocast():
            return super()._reference_forward(batch, trace_blocks=trace_blocks)

    def _configure_trainability(self, *, stage: str, train: bool) -> None:
        super()._configure_trainability(stage=stage, train=train)
        if not self.trainer_config.finetune_backbone:
            self.model.model.backbone.requires_grad_(False)
            self.model.model.backbone.eval()

    def _clip_trainable_gradients(self) -> None:
        max_norm = float(self.trainer_config.grad_clip_norm)
        if max_norm <= 0:
            return
        parameters = [
            parameter
            for module in (self.model, self.completion_module)
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def _optimizer_step(self, loss: torch.Tensor, *, train: bool) -> None:
        if not train or not loss.requires_grad:
            return
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is None:
            loss.backward()
            self._clip_trainable_gradients()
            self.optimizer.step()
            return
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self._clip_trainable_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()
