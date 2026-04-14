"""Task callback protocol and minimal reusable callbacks."""

from elastic_method.tasks.protocol import TaskCallback
from elastic_method.tasks.regression import MeanPooledRegressionCallback

__all__ = ["MeanPooledRegressionCallback", "TaskCallback"]
