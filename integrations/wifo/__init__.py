"""WiFo integration for StruJEPA using the benchmark-owned WiFo backbone."""

from integrations.wifo.benchmark_paths import (
    WifoBenchmarkPaths,
    ensure_wifo_source_path,
    resolve_wifo_benchmark_paths,
)
from integrations.wifo.elastic_wifo import (
    build_headwise_width_multipliers,
    build_layerwise_depth_multipliers,
    elasticize_wifo,
)
from integrations.wifo.strujepa_recipe_trainer import WiFoStruJEPATrainer
from integrations.wifo.strujepa_wifo import WiFoTaskAdapter

__all__ = [
    "WiFoBenchmarkPaths",
    "WiFoTaskAdapter",
    "WiFoStruJEPATrainer",
    "build_headwise_width_multipliers",
    "build_layerwise_depth_multipliers",
    "elasticize_wifo",
    "ensure_wifo_source_path",
    "resolve_wifo_benchmark_paths",
]
