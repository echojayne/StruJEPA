"""Core runtime, elasticization, and data structures."""

from elastic_method.core.elasticizer import elasticize_model
from elastic_method.core.structures import (
    ElasticStackMetadata,
    ElasticSubnet,
    ElasticizationSpec,
    ForwardResult,
    StructureMaskDescriptor,
)
from elastic_method.core.subnet import (
    dedupe_subnets,
    resolve_active_ffn,
    resolve_active_heads,
    resolve_active_layers,
    select_depth_indices,
)

__all__ = [
    "ElasticStackMetadata",
    "ElasticSubnet",
    "ElasticizationSpec",
    "ForwardResult",
    "StructureMaskDescriptor",
    "dedupe_subnets",
    "elasticize_model",
    "resolve_active_ffn",
    "resolve_active_heads",
    "resolve_active_layers",
    "select_depth_indices",
]
