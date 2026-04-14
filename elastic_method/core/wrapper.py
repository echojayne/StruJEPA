"""Whole-model wrapper that routes width/depth controls into elastic blocks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from torch import nn

from elastic_method.adapters.base import BlockAdapter
from elastic_method.core.runtime import ElasticRuntimeState, elastic_runtime
from elastic_method.core.structures import (
    ElasticStackMetadata,
    ElasticizationSpec,
    ForwardResult,
    StructureMaskDescriptor,
)
from elastic_method.core.subnet import (
    resolve_active_ffn,
    resolve_active_heads,
    resolve_active_layers,
    select_depth_indices,
)


class ElasticModelWrapper(nn.Module):
    """Wrap a model and an explicit encoder stack path with elastic controls."""

    def __init__(self, model: nn.Module, *, spec: ElasticizationSpec, adapter: BlockAdapter) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.adapter = adapter
        self.stack_handle = self.adapter.elasticize(model, stack_path=spec.stack_path)
        self.metadata: ElasticStackMetadata = self.stack_handle.metadata

    def _build_structure_mask(
        self,
        *,
        width_multiplier: float,
        depth_multiplier: float,
    ) -> StructureMaskDescriptor:
        active_heads = resolve_active_heads(
            max_heads=self.metadata.max_num_heads,
            width_multiplier=width_multiplier,
        )
        active_ffn = resolve_active_ffn(
            max_ffn_dim=self.metadata.max_ffn_dim,
            width_multiplier=width_multiplier,
        )
        active_layers = resolve_active_layers(
            max_layers=self.metadata.total_layers,
            depth_multiplier=depth_multiplier,
        )
        selected_layers = tuple(
            select_depth_indices(total_layers=self.metadata.total_layers, active_layers=active_layers)
        )
        return StructureMaskDescriptor(
            width_multiplier=float(width_multiplier),
            depth_multiplier=float(depth_multiplier),
            total_layers=int(self.metadata.total_layers),
            selected_layer_indices=selected_layers,
            active_num_heads=int(active_heads),
            active_ffn_dim=int(active_ffn),
        )

    def forward(
        self,
        *args: Any,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        return_encoder_state: bool = False,
        **kwargs: Any,
    ) -> ForwardResult:
        structure_mask = self._build_structure_mask(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
        )
        runtime = ElasticRuntimeState(
            width_multiplier=float(width_multiplier),
            depth_multiplier=float(depth_multiplier),
            selected_layer_indices=structure_mask.selected_layer_indices,
            active_num_heads=structure_mask.active_num_heads,
            active_ffn_dim=structure_mask.active_ffn_dim,
            return_encoder_state=bool(return_encoder_state),
        )
        with elastic_runtime(runtime):
            model_output = self.model(*args, **kwargs)
        aux = {"stack_path": self.spec.stack_path, "block_family": self.spec.block_family}
        aux.update(asdict(self.metadata))
        return ForwardResult(
            model_output=model_output,
            encoder_state=runtime.last_encoder_state if return_encoder_state else None,
            structure_mask=structure_mask,
            aux=aux,
        )
