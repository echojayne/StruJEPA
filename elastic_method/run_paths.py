"""Path helpers for keeping StruJEPA benchmark runs outside the source tree."""

from __future__ import annotations

import os
from pathlib import Path


RUN_ROOT_ENV = "STRUJEPA_RAN_RUN_ROOT"
DEFAULT_RUN_WORKSPACE = "strujepa_ran_runs"


def strujepa_run_root(project_root: str | Path | None = None) -> Path:
    """Return the external workspace used for StruJEPA benchmark run outputs."""

    raw_root = os.environ.get(RUN_ROOT_ENV)
    if raw_root:
        return Path(os.path.expandvars(raw_root)).expanduser()
    if project_root is None:
        project_root = Path.cwd()
    return Path(project_root).expanduser().resolve().parent / DEFAULT_RUN_WORKSPACE


def resolve_run_output_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve launch output paths while redirecting relative runs/ paths."""

    expanded = Path(os.path.expandvars(str(path))).expanduser()
    if expanded.is_absolute():
        return expanded
    if expanded.parts and expanded.parts[0] == "runs":
        return strujepa_run_root(project_root) / expanded
    return expanded
