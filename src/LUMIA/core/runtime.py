"""Canonical deterministic PyTorch runtime configuration and device routing.

Single source of truth shared by Module 4 (``LUMIA.physics.gpu_processing``) and
Module 7 (``LUMIA.ml``). Contains no persistence, orchestration, or HTTP dependencies.
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import torch

__all__ = ["DEVICE_SPEC_PATTERN", "configure_deterministic_torch", "resolve_torch_device"]

DEVICE_SPEC_PATTERN: str = r"^(cpu|cuda|auto|cuda:\d+)$"
"""Regex accepted by every Pydantic ``device`` field: cpu, cuda, cuda:N or auto."""

_CUDA_INDEXED_PATTERN: re.Pattern[str] = re.compile(r"^cuda:(\d+)$")


def configure_deterministic_torch(
    seed: int = 42,
    *,
    deterministic: bool = True,
    num_cpu_threads: int | None = None,
) -> None:
    """Seed Python/NumPy/PyTorch RNGs, enforce deterministic kernels, and set CPU threads.

    When ``deterministic`` is True this activates ``torch.use_deterministic_algorithms``,
    forces deterministic cuDNN convolutions and disables cuDNN benchmark auto-tuning.

    Callers should resolve the target device with :func:`resolve_torch_device` *before*
    calling this function so that an unavailable GPU fails without mutating global state.
    """
    if seed < 0:
        msg = f"seed must be non-negative, got {seed}."
        raise ValueError(msg)

    if num_cpu_threads is not None:
        if num_cpu_threads < 1:
            msg = f"num_cpu_threads must be at least 1, got {num_cpu_threads}."
            raise ValueError(msg)
        torch.set_num_threads(num_cpu_threads)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for third-party libraries
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def resolve_torch_device(device_spec: str) -> torch.device:
    """Resolve ``'cpu'``, ``'cuda'``, ``'cuda:N'`` or ``'auto'`` to a validated device.

    Raises:
        RuntimeError: If a CUDA device is requested but unavailable.
        ValueError: If ``device_spec`` is not a recognized device specifier.
    """
    if device_spec == "cpu":
        return torch.device("cpu")
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA device requested, but torch.cuda.is_available() is False."
            raise RuntimeError(msg)
        return torch.device("cuda")

    match = _CUDA_INDEXED_PATTERN.match(device_spec)
    if match is not None:
        index = int(match.group(1))
        if not torch.cuda.is_available():
            msg = f"CUDA device '{device_spec}' requested, but CUDA is not available."
            raise RuntimeError(msg)
        if index >= torch.cuda.device_count():
            msg = (
                f"CUDA device index {index} out of range "
                f"(available devices: {torch.cuda.device_count()})."
            )
            raise RuntimeError(msg)
        return torch.device("cuda", index)

    msg = f"Unrecognized device specifier '{device_spec}'."
    raise ValueError(msg)
