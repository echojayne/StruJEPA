"""Public API for the current StruJEPA method."""

from elastic_method.core.elasticizer import elasticize_model
from elastic_method.core.structures import (
    ElasticizationSpec,
    ForwardResult,
    StructureMaskDescriptor,
)
from elastic_method.method.completion import (
    CompletionStageEpochs,
    WidthOperatorCompletionConfig,
)
from elastic_method.method.trainer import StruJEPATrainer, MethodConfig
from elastic_method.tasks.protocol import TaskAdapter

__all__ = [
    "StruJEPATrainer",
    "CompletionStageEpochs",
    "ElasticizationSpec",
    "ForwardResult",
    "MethodConfig",
    "StructureMaskDescriptor",
    "TaskAdapter",
    "WidthOperatorCompletionConfig",
    "elasticize_model",
]
