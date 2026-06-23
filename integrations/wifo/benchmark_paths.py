from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_ROOT = Path.home() / "ai_ran_benchmarks"


@dataclass(frozen=True)
class WifoBenchmarkPaths:
    root: Path
    source_root: Path
    train_val_data: Path
    test_data: Path
    original_weights: Path

    @property
    def source_src(self) -> Path:
        return self.source_root / "src"


def _default_paths(root: Path) -> WifoBenchmarkPaths:
    return WifoBenchmarkPaths(
        root=root,
        source_root=root / "source_snapshots" / "wifo",
        train_val_data=root / "benchmarks" / "channel_prediction" / "wifo" / "assets" / "train_val_data",
        test_data=root / "benchmarks" / "channel_prediction" / "wifo" / "assets" / "test_data",
        original_weights=root / "benchmarks" / "channel_prediction" / "wifo" / "assets" / "original_weights",
    )


def _load_catalog(catalog_path: Path) -> dict[str, Any] | None:
    if not catalog_path.exists():
        return None
    try:
        import yaml
    except Exception:
        return None
    payload = yaml.safe_load(catalog_path.read_text())
    return payload if isinstance(payload, dict) else None


def resolve_wifo_benchmark_paths(root: str | Path | None = None) -> WifoBenchmarkPaths:
    resolved_root = Path(root or os.environ.get("AI_RAN_BENCHMARK_ROOT", DEFAULT_BENCHMARK_ROOT)).expanduser().resolve()
    defaults = _default_paths(resolved_root)
    catalog = _load_catalog(resolved_root / "catalog" / "benchmark_catalog.yaml")
    if catalog is None:
        return defaults

    try:
        wifo = catalog["tasks"]["channel_prediction"]["models"]["wifo"]
        assets = wifo["assets"]
        return WifoBenchmarkPaths(
            root=resolved_root,
            source_root=Path(wifo.get("source_root", defaults.source_root)).expanduser().resolve(),
            train_val_data=Path(assets.get("train_val_data", defaults.train_val_data)).expanduser().resolve(),
            test_data=Path(assets.get("test_data", defaults.test_data)).expanduser().resolve(),
            original_weights=Path(assets.get("original_weights", defaults.original_weights)).expanduser().resolve(),
        )
    except Exception:
        return defaults


def ensure_wifo_source_path(paths: WifoBenchmarkPaths | None = None) -> Path:
    paths = paths or resolve_wifo_benchmark_paths()
    source_src = paths.source_src
    if str(source_src) not in sys.path:
        sys.path.insert(0, str(source_src))
    return source_src
