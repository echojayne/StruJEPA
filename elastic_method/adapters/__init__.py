"""Block-family adapters for elasticizing standard encoder stacks."""

from elastic_method.adapters.base import BlockAdapter, ElasticizedStackHandle
from elastic_method.adapters.registry import get_block_adapter, register_block_adapter

__all__ = [
    "BlockAdapter",
    "ElasticizedStackHandle",
    "get_block_adapter",
    "register_block_adapter",
]

# Import side effects register supported adapters.
from elastic_method.adapters import hf_bert, hf_vit, timm_vit, torch_encoder, wifo_vit  # noqa: E402,F401
