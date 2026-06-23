"""Block-family adapters for elasticizing standard encoder stacks."""

import importlib
import importlib.util

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.registry import get_block_adapter, register_block_adapter

__all__ = [
    "BlockAdapter",
    "ElasticizedStackHandle",
    "get_block_adapter",
    "register_block_adapter",
]

# Import side effects register supported adapters.
from elastic_method.adapters import perceiver_io, torch_encoder, wifo_vit  # noqa: E402,F401


def _try_register_optional_adapter(name: str, dependencies: tuple[str, ...] = ()) -> None:
    if any(importlib.util.find_spec(dependency) is None for dependency in dependencies):
        return
    try:
        importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        return


for _adapter_name, _dependencies in (
    ("dinov3_vit", ("transformers", "tqdm")),
    ("hf_bert", ("transformers", "tqdm")),
    ("hf_vit", ("transformers", "tqdm")),
    ("timm_vit", ("timm",)),
):
    _try_register_optional_adapter(_adapter_name, _dependencies)
