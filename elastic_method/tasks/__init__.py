"""Task adapter protocol and minimal reusable adapters."""

from elastic_method.tasks.protocol import TaskAdapter
from elastic_method.tasks.regression import MeanPooledRegressionTaskAdapter

__all__ = ["MeanPooledRegressionTaskAdapter", "TaskAdapter"]
