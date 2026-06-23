"""Current width-operator completion training method."""

from elastic_method.method.completion import (
    CompletionStageEpochs,
    WidthOperatorCompletionConfig,
)
from elastic_method.method.trainer import StruJEPATrainer, MethodConfig

__all__ = [
    "StruJEPATrainer",
    "CompletionStageEpochs",
    "MethodConfig",
    "WidthOperatorCompletionConfig",
]
