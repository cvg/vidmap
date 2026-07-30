"""Lightweight deterministic frontend settings used before numerical imports."""

from typing import Any

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def deterministic_env(seed: int = 0) -> dict[str, str]:
    return {
        **THREAD_ENV,
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def deterministic_config_patch() -> dict[str, Any]:
    return {
        "roma": {"compile": False},
        "keyframes": {"matching": {"batch_size": 1, "num_workers": 0}},
        "tracks": {"propagation": {"max_sequential_track_sigma_roma_px": None}},
        "depth": {
            "batch_size": 1,
            "num_workers": 0,
            "cache_map": False,
        },
    }
