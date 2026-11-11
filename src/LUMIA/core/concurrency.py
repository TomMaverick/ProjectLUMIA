"""Central runtime resource manager and thread budgeting controller for Project LUMIA.

Coordinates global CPU thread limits across PyTorch, NumPy, OpenMP, MKL, and BLAS
backends, and manages VRAM allocation ceilings to prevent CPU oversubscription.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from LUMIA.core.config import ResourceBudgetConfig

__all__ = [
    "ResourceBudgetConfig",
    "configure_global_runtime",
    "get_active_resource_budget",
]

_ACTIVE_RESOURCE_BUDGET: ResourceBudgetConfig | None = None

_THREADING_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def get_active_resource_budget() -> ResourceBudgetConfig | None:
    """Return the currently active global resource budget, or None if unconfigured."""
    return _ACTIVE_RESOURCE_BUDGET


def configure_global_runtime(
    config: ResourceBudgetConfig | None = None,
) -> ResourceBudgetConfig:
    """Enforce central resource budgeting across PyTorch, BLAS, OpenMP, and OpenCV.

    Parameters:
        config: Explicit budget configuration. If None, constructed from environment.

    Returns:
        The validated active ResourceBudgetConfig.
    """
    global _ACTIVE_RESOURCE_BUDGET

    resolved_config = config if config is not None else ResourceBudgetConfig.from_env()

    # 1. PyTorch CPU intra-op thread limit
    torch.set_num_threads(resolved_config.num_cpu_threads)

    # 2. PyTorch inter-op threads (can only be configured before operations run)
    try:
        interop_threads = max(1, min(4, resolved_config.num_cpu_threads))
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    # 3. OpenMP / MKL / BLAS environment variable limits
    if resolved_config.enable_openmp_threading_control:
        threads_str = str(resolved_config.num_cpu_threads)
        for var_name in _THREADING_ENV_VARS:
            os.environ[var_name] = threads_str

    # 4. OpenCV thread limits (if cv2 is installed)
    if "cv2" in sys.modules:
        cv2 = sys.modules["cv2"]
        if hasattr(cv2, "setNumThreads"):
            cv2.setNumThreads(resolved_config.num_cpu_threads)
    else:
        try:
            import cv2
            cv2.setNumThreads(resolved_config.num_cpu_threads)
        except ImportError:
            pass

    # 5. PyTorch CUDA memory fraction limiting
    if torch.cuda.is_available() and resolved_config.max_gpu_memory_gb is not None:
        max_bytes = resolved_config.max_gpu_memory_gb * (1024**3)
        for i in range(torch.cuda.device_count()):
            try:
                total_bytes = torch.cuda.get_device_properties(i).total_memory
                fraction = min(1.0, max(0.01, max_bytes / total_bytes))
                torch.cuda.set_per_process_memory_fraction(fraction, device=i)
            except (RuntimeError, ValueError):
                pass

    # 6. Ensure telescope FITS archive directory exists
    try:
        Path(resolved_config.archive_storage_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    _ACTIVE_RESOURCE_BUDGET = resolved_config
    return resolved_config
