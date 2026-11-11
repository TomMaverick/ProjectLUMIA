"""Canonical resource budgeting and runtime configuration models for Project LUMIA."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field

from LUMIA.core.schemas import LumiaBaseModel

__all__ = ["ResourceBudgetConfig"]


def _default_cpu_threads() -> int:
    """Default to logical CPU cores minus 1, with a minimum of 1 thread."""
    cores = os.cpu_count() or 1
    return max(1, cores - 1)


class ResourceBudgetConfig(LumiaBaseModel):
    """Central validated resource and CPU/GPU thread budgeting configuration."""

    num_cpu_threads: int = Field(
        default_factory=_default_cpu_threads,
        ge=1,
        description="Number of CPU worker threads allocated for computation (defaults to logical cores minus 1).",
    )
    max_gpu_memory_gb: float = Field(
        default=4.0,
        gt=0.0,
        description="Maximum VRAM allocation in gigabytes for PyTorch GPU operations.",
    )
    enable_openmp_threading_control: bool = Field(
        default=True,
        description="Whether OpenMP, MKL, OpenBLAS, and NumExpr thread limits are globally enforced.",
    )
    archive_storage_path: str = Field(
        default="data/archives/",
        min_length=1,
        description="Filesystem directory path for raw and calibrated telescope FITS archives.",
    )

    @classmethod
    def from_env(cls) -> ResourceBudgetConfig:
        """Construct ResourceBudgetConfig from LUMIA_* environment variables."""
        kwargs: dict[str, Any] = {}

        env_threads = os.environ.get("LUMIA_NUM_CPU_THREADS")
        if env_threads is not None and env_threads.strip():
            kwargs["num_cpu_threads"] = int(env_threads.strip())

        env_gpu_mem = os.environ.get("LUMIA_MAX_GPU_MEMORY_GB")
        if env_gpu_mem is not None and env_gpu_mem.strip():
            kwargs["max_gpu_memory_gb"] = float(env_gpu_mem.strip())

        env_openmp = os.environ.get("LUMIA_ENABLE_OPENMP_THREADING_CONTROL")
        if env_openmp is not None and env_openmp.strip():
            kwargs["enable_openmp_threading_control"] = env_openmp.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        env_archive = os.environ.get("LUMIA_ARCHIVE_STORAGE_PATH")
        if env_archive is not None and env_archive.strip():
            kwargs["archive_storage_path"] = env_archive.strip()

        return cls(**kwargs)
