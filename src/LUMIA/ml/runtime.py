"""Module 7 runtime facade re-exporting the canonical ``LUMIA.core.runtime`` utilities."""

from __future__ import annotations

from LUMIA.core.runtime import configure_deterministic_torch, resolve_torch_device

__all__ = ["configure_deterministic_torch", "resolve_torch_device"]
