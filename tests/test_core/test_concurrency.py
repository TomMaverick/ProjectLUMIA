"""Hermetic unit tests for central resource budgeting and runtime thread manager."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from LUMIA.api.main import create_app
from LUMIA.core.concurrency import (
    ResourceBudgetConfig,
    configure_global_runtime,
    get_active_resource_budget,
)
from LUMIA.core.runtime import configure_deterministic_torch
from LUMIA.physics.gpu_processing import GPUProcessingConfig


class TestResourceBudgetConfig:
    """Validate Pydantic contract and environment-based instantiation."""

    def test_default_resource_budget_values(self) -> None:
        """Default configuration adopts system core count minus 1 (min 1)."""
        config = ResourceBudgetConfig()
        expected_min_threads = max(1, (os.cpu_count() or 1) - 1)

        assert config.num_cpu_threads == expected_min_threads
        assert config.num_cpu_threads >= 1
        assert pytest.approx(config.max_gpu_memory_gb) == 4.0
        assert config.enable_openmp_threading_control is True

    def test_custom_valid_resource_budget(self) -> None:
        """Explicit valid parameter assignments succeed."""
        config = ResourceBudgetConfig(
            num_cpu_threads=4,
            max_gpu_memory_gb=8.5,
            enable_openmp_threading_control=False,
        )
        assert config.num_cpu_threads == 4
        assert pytest.approx(config.max_gpu_memory_gb) == 8.5
        assert config.enable_openmp_threading_control is False

    @pytest.mark.parametrize("invalid_threads", [0, -1, -10])
    def test_non_positive_cpu_threads_raises_validation_error(
        self, invalid_threads: int
    ) -> None:
        """Thread counts below 1 must be rejected by Pydantic."""
        with pytest.raises(ValidationError):
            ResourceBudgetConfig(num_cpu_threads=invalid_threads)

    @pytest.mark.parametrize("invalid_gpu_mem", [0.0, -1.0, -50.0])
    def test_non_positive_gpu_memory_raises_validation_error(
        self, invalid_gpu_mem: float
    ) -> None:
        """VRAM limits at or below zero must be rejected by Pydantic."""
        with pytest.raises(ValidationError):
            ResourceBudgetConfig(max_gpu_memory_gb=invalid_gpu_mem)

    def test_from_env_parses_environment_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables LUMIA_* are correctly parsed into the configuration."""
        monkeypatch.setenv("LUMIA_NUM_CPU_THREADS", "6")
        monkeypatch.setenv("LUMIA_MAX_GPU_MEMORY_GB", "12.0")
        monkeypatch.setenv("LUMIA_ENABLE_OPENMP_THREADING_CONTROL", "false")

        config = ResourceBudgetConfig.from_env()
        assert config.num_cpu_threads == 6
        assert pytest.approx(config.max_gpu_memory_gb) == 12.0
        assert config.enable_openmp_threading_control is False

    def test_from_env_adopts_defaults_when_env_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty environment variables fall back to baseline defaults."""
        monkeypatch.delenv("LUMIA_NUM_CPU_THREADS", raising=False)
        monkeypatch.delenv("LUMIA_MAX_GPU_MEMORY_GB", raising=False)
        monkeypatch.delenv("LUMIA_ENABLE_OPENMP_THREADING_CONTROL", raising=False)

        config = ResourceBudgetConfig.from_env()
        assert config.num_cpu_threads >= 1
        assert pytest.approx(config.max_gpu_memory_gb) == 4.0
        assert config.enable_openmp_threading_control is True


class TestConfigureGlobalRuntime:
    """Verify thread limit propagation to PyTorch, OpenMP, and BLAS environments."""

    def test_configure_global_runtime_sets_pytorch_threads(self) -> None:
        """PyTorch intra-op thread count is updated to configured thread count."""
        budget = ResourceBudgetConfig(
            num_cpu_threads=3,
            enable_openmp_threading_control=True,
        )
        applied = configure_global_runtime(budget)

        assert applied.num_cpu_threads == 3
        assert torch.get_num_threads() == 3
        assert get_active_resource_budget() is applied

    def test_configure_global_runtime_sets_openmp_env_vars(self) -> None:
        """OpenMP, MKL, OpenBLAS, and NumExpr environment variables are bound."""
        budget = ResourceBudgetConfig(
            num_cpu_threads=2,
            enable_openmp_threading_control=True,
        )
        configure_global_runtime(budget)

        assert os.environ.get("OMP_NUM_THREADS") == "2"
        assert os.environ.get("MKL_NUM_THREADS") == "2"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "2"
        assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "2"

    def test_configure_global_runtime_skips_env_vars_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When enable_openmp_threading_control=False, env vars remain untouched."""
        monkeypatch.setenv("OMP_NUM_THREADS", "99")
        budget = ResourceBudgetConfig(
            num_cpu_threads=4,
            enable_openmp_threading_control=False,
        )
        configure_global_runtime(budget)

        assert os.environ.get("OMP_NUM_THREADS") == "99"

    def test_configure_global_runtime_gpu_fraction_mock(self) -> None:
        """When CUDA is available, set_per_process_memory_fraction is called."""
        budget = ResourceBudgetConfig(num_cpu_threads=2, max_gpu_memory_gb=6.0)

        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024**3)  # 12 GB VRAM

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.set_per_process_memory_fraction") as mock_set_fraction,
        ):
            configure_global_runtime(budget)
            mock_set_fraction.assert_called_once_with(0.5, device=0)


class TestRuntimeDeterministicTorchThreadIntegration:
    """Verify configure_deterministic_torch supports num_cpu_threads."""

    def test_deterministic_torch_with_threads(self) -> None:
        """configure_deterministic_torch successfully binds thread limit."""
        configure_deterministic_torch(seed=123, deterministic=True, num_cpu_threads=2)
        assert torch.get_num_threads() == 2

    @pytest.mark.parametrize("invalid_threads", [0, -1])
    def test_deterministic_torch_invalid_threads_raises_value_error(
        self, invalid_threads: int
    ) -> None:
        """Non-positive thread counts in configure_deterministic_torch raise ValueError."""
        with pytest.raises(ValueError, match="num_cpu_threads must be at least 1"):
            configure_deterministic_torch(seed=42, num_cpu_threads=invalid_threads)


class TestGPUProcessingConfigThreadLimit:
    """Verify GPUProcessingConfig accepts and validates num_cpu_threads."""

    def test_gpu_processing_config_num_cpu_threads_valid(self) -> None:
        """Valid num_cpu_threads passes validation."""
        cfg = GPUProcessingConfig(num_cpu_threads=4)
        assert cfg.num_cpu_threads == 4

    @pytest.mark.parametrize("invalid_threads", [0, -2])
    def test_gpu_processing_config_invalid_threads_raises(
        self, invalid_threads: int
    ) -> None:
        """Non-positive num_cpu_threads raises ValidationError."""
        with pytest.raises(ValidationError):
            GPUProcessingConfig(num_cpu_threads=invalid_threads)


class TestFastAPIAppResourceBudgetIntegration:
    """Verify FastAPI application factory accepts and stores resource budget."""

    def test_create_app_with_explicit_resource_budget(self) -> None:
        """create_app accepts and assigns ResourceBudgetConfig to app.state."""
        budget = ResourceBudgetConfig(num_cpu_threads=2)
        app = create_app(resource_budget=budget)

        assert hasattr(app.state, "resource_budget")
        assert app.state.resource_budget.num_cpu_threads == 2
