"""Public entry point for building an isolated elastic model wrapper."""

from __future__ import annotations

from copy import deepcopy

from torch import nn

from elastic_method.adapters import get_block_adapter
from elastic_method.core.structures import ElasticizationSpec
from elastic_method.core.wrapper import ElasticModelWrapper


def elasticize_model(
    model: nn.Module,
    spec: ElasticizationSpec,
    *,
    copy_model: bool = True,
) -> ElasticModelWrapper:
    model_copy = deepcopy(model) if copy_model else model
    adapter = get_block_adapter(spec.block_family)
    return ElasticModelWrapper(model_copy, spec=spec, adapter=adapter)
