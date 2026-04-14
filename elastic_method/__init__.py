"""Isolated elastic-method framework for standard Transformer encoder stacks."""

from elastic_method.core.elasticizer import elasticize_model
from elastic_method.core.structures import (
    ElasticizationSpec,
    ForwardResult,
    StructureMaskDescriptor,
)
from elastic_method.method.mask import (
    RepresentationAlignmentModule,
    StructuralMaskEncoder,
    StructuralMaskModule,
)
from elastic_method.method.trainer import AlignmentTrainer, MethodConfig

__all__ = [
    "AlignmentTrainer",
    "ElasticizationSpec",
    "ForwardResult",
    "MethodConfig",
    "RepresentationAlignmentModule",
    "StructuralMaskEncoder",
    "StructuralMaskModule",
    "StructureMaskDescriptor",
    "elasticize_model",
]
