"""Unified method components for alignment-based elastic training."""

from elastic_method.method.mask import (
    RepresentationAlignmentModule,
    StructuralMaskEncoder,
    StructuralMaskModule,
)
from elastic_method.method.trainer import AlignmentTrainer, MethodConfig

__all__ = [
    "AlignmentTrainer",
    "MethodConfig",
    "RepresentationAlignmentModule",
    "StructuralMaskEncoder",
    "StructuralMaskModule",
]
