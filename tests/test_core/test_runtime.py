"""P1-1: single canonical PyTorch runtime shared by Module 4 (physics) and Module 7 (ml)."""

from __future__ import annotations

import ast
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import LUMIA.ml.classifier as ml_classifier
import LUMIA.ml.runtime as ml_runtime
import LUMIA.physics.gpu_processing as gpu_processing
from LUMIA.core import runtime as core_runtime
from LUMIA.physics.gpu_processing import GPUProcessingConfig, shift_and_stack

_SRC_ROOT: Path = Path(__file__).resolve().parents[2] / "src" / "LUMIA"


def test_modules_4_and_7_share_the_canonical_runtime_objects() -> None:
    """Both modules must expose the exact same function objects as core.runtime."""
    for name in ("configure_deterministic_torch", "resolve_torch_device"):
        canonical = getattr(core_runtime, name)
        assert getattr(gpu_processing, name) is canonical
        assert getattr(ml_runtime, name) is canonical
        assert getattr(ml_classifier, name) is canonical


def test_gpu_processing_defines_no_local_runtime_helpers() -> None:
    """AST guard: gpu_processing.py must not re-implement the runtime helpers."""
    tree = ast.parse((_SRC_ROOT / "physics" / "gpu_processing.py").read_text(encoding="utf-8"))
    local_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "configure_deterministic_torch" not in local_functions
    assert "resolve_torch_device" not in local_functions


def test_ml_runtime_defines_no_local_runtime_helpers() -> None:
    """AST guard: ml/runtime.py is at most a thin re-export of core.runtime."""
    tree = ast.parse((_SRC_ROOT / "ml" / "runtime.py").read_text(encoding="utf-8"))
    local_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert local_functions == set()


def test_configure_deterministic_torch_seeds_all_rngs_and_sets_flags() -> None:
    """Python, NumPy and PyTorch RNGs are seeded and deterministic flags are enforced."""
    torch.use_deterministic_algorithms(False)
    core_runtime.configure_deterministic_torch(2024, deterministic=True)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))  # noqa: NPY002

    core_runtime.configure_deterministic_torch(2024, deterministic=True)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))  # noqa: NPY002

    assert first == second
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_configure_deterministic_torch_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        core_runtime.configure_deterministic_torch(-1)


def test_resolve_torch_device_supports_indexed_and_rejects_unknown_specs() -> None:
    assert core_runtime.resolve_torch_device("cpu").type == "cpu"
    assert core_runtime.resolve_torch_device("auto").type in {"cpu", "cuda"}
    with pytest.raises(ValueError, match="device"):
        core_runtime.resolve_torch_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            core_runtime.resolve_torch_device("cuda:0")


def test_gpu_processing_config_accepts_indexed_cuda_devices() -> None:
    """Module 4 config shares Module 7's device grammar (cuda:N) via the common runtime."""
    assert GPUProcessingConfig(device="cuda:1").device == "cuda:1"
    with pytest.raises(ValidationError):
        GPUProcessingConfig(device="gpu0")


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CUDA-less host")
def test_device_is_resolved_before_global_state_is_mutated() -> None:
    """An unavailable GPU must fail before RNG/determinism flags are touched (audit 2.5.2)."""
    torch.use_deterministic_algorithms(False)
    assert not torch.are_deterministic_algorithms_enabled()

    frames = np.zeros((3, 16, 16), dtype=np.float32)
    times = np.arange(3, dtype=np.float32)
    velocities = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="CUDA"):
        shift_and_stack(
            frames,
            times,
            velocities,
            config=GPUProcessingConfig(device="cuda", deterministic=True),
        )

    assert not torch.are_deterministic_algorithms_enabled()
