from __future__ import annotations

from pathlib import Path


def load_mat_file(path: str | Path):
    path = Path(path)
    load_errors: list[str] = []

    try:
        import hdf5storage

        return hdf5storage.loadmat(str(path))
    except Exception as exc:
        load_errors.append(f"hdf5storage: {exc}")

    try:
        from scipy.io import loadmat

        return loadmat(str(path))
    except Exception as exc:
        load_errors.append(f"scipy.io.loadmat: {exc}")

    joined = "; ".join(load_errors) if load_errors else "no backend attempted"
    raise ImportError(
        f"Unable to load MAT file '{path}'. Install 'hdf5storage' for MATLAB v7.3 files "
        f"or 'scipy' for v5 MAT files. Backend errors: {joined}"
    )


def extract_first_array(mat_data, preferred_keys: tuple[str, ...]):
    for key in preferred_keys:
        if key in mat_data:
            return mat_data[key]
    for key, value in mat_data.items():
        if not str(key).startswith("__"):
            return value
    raise KeyError(f"Unable to find a non-metadata array. Looked for keys: {preferred_keys}")
