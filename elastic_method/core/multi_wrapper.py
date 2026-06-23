"""Elastic wrapper for models with more than one torch TransformerEncoder stack."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from torch import nn

from elastic_method.adapters.common import get_target_module, replace_stack_blocks, resolve_stack_sequence
from elastic_method.adapters.torch_encoder import ElasticTorchEncoderLayer
from elastic_method.core.runtime import ElasticRuntimeState, elastic_runtime
from elastic_method.core.structures import ElasticStackMetadata, ForwardResult, StructureMaskDescriptor
from elastic_method.core.subnet import (
    resolve_active_ffn,
    resolve_active_heads,
    resolve_active_layers,
    select_depth_indices,
)


class MultiStackTorchEncoderWrapper(nn.Module):
    """Route one width/depth subnet descriptor through multiple torch encoder stacks.

    A-MMSE has separate frequency and temporal TransformerEncoder stacks. Treating
    them as one global elastic depth keeps the StruJEPA subnet descriptor identical
    to the single-stack case while still skipping layers in both stacks.
    """

    def __init__(self, model: nn.Module, *, stack_paths: tuple[str, ...]) -> None:
        super().__init__()
        if not stack_paths:
            raise ValueError("stack_paths must not be empty")
        self.model = model
        self.stack_paths = tuple(str(path) for path in stack_paths)

        stack_layers: list[tuple[str, nn.TransformerEncoderLayer]] = []
        for stack_path in self.stack_paths:
            target = get_target_module(model, stack_path)
            sequence = resolve_stack_sequence(target)
            layers = list(sequence)
            if not layers:
                raise ValueError(f"stack at '{stack_path}' is empty")
            for layer in layers:
                if not isinstance(layer, nn.TransformerEncoderLayer):
                    raise TypeError(
                        "MultiStackTorchEncoderWrapper only supports nn.TransformerEncoderLayer stacks"
                    )
                stack_layers.append((stack_path, layer))

        total_layers = len(stack_layers)
        global_index = 0
        self.blocks: list[nn.Module] = []
        self.stack_layer_ranges: list[tuple[int, int]] = []
        for stack_path in self.stack_paths:
            target = get_target_module(model, stack_path)
            sequence = resolve_stack_sequence(target)
            original_layers = list(sequence)
            elastic_layers: list[nn.Module] = []
            start_index = global_index
            for layer in original_layers:
                elastic_layers.append(
                    ElasticTorchEncoderLayer(
                        layer,
                        layer_index=global_index,
                        total_layers=total_layers,
                    )
                )
                global_index += 1
            self.stack_layer_ranges.append((start_index, global_index))
            _, replaced = replace_stack_blocks(target, elastic_layers)
            self.blocks.extend(replaced)

        first_block = self.blocks[0]
        self.metadata = ElasticStackMetadata(
            family="torch_encoder_multi",
            total_layers=total_layers,
            max_num_heads=int(first_block.max_num_heads),
            max_ffn_dim=int(first_block.max_ffn_dim),
        )

    def _build_structure_mask(self, *, width_multiplier: float, depth_multiplier: float) -> StructureMaskDescriptor:
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
        if os.environ.get("STRUJEPA_DEPTH_SELECTION", "").strip().lower() == "ofa_stage_prefix":
            selected_layers = tuple(
                layer_index
                for start, end in self.stack_layer_ranges
                for layer_index in range(
                    start,
                    start
                    + resolve_active_layers(
                        max_layers=end - start,
                        depth_multiplier=depth_multiplier,
                    ),
                )
            )
        else:
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
        completion_module: nn.Module | None = None,
        trace_blocks: bool = False,
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
            completion_module=completion_module,
            trace_blocks=bool(trace_blocks),
        )
        with elastic_runtime(runtime):
            model_output = self.model(*args, **kwargs)
        aux = {"stack_paths": self.stack_paths, "block_family": self.metadata.family}
        aux.update(asdict(self.metadata))
        aux["block_traces"] = runtime.block_traces
        aux["completion_losses"] = runtime.completion_losses
        return ForwardResult(
            model_output=model_output,
            encoder_state=runtime.last_encoder_state if return_encoder_state else None,
            structure_mask=structure_mask,
            aux=aux,
        )
