"""Project LUMIA: Low-SNR Unidentified Moving Interplanetary Analyzer."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from LUMIA.core.schemas import CandidateDetection

if TYPE_CHECKING:
    from types import ModuleType

    from LUMIA import api, core, data_pipeline, ml, physics

__version__: str = "0.2.0"

_SUBPACKAGES: frozenset[str] = frozenset(
    {
        "api",
        "core",
        "data_mining",
        "data_pipeline",
        "ml",
        "models",
        "physics",
        "training",
    }
)

__all__ = [
    "CandidateDetection",
    "__version__",
    "api",
    "core",
    "data_mining",
    "data_pipeline",
    "ml",
    "models",
    "physics",
    "training",
]


def __getattr__(name: str) -> ModuleType:
    """Lazily import top-level LUMIA subpackages on first attribute access."""
    if name in _SUBPACKAGES:
        package_name = __name__.removesuffix(".__init__")
        module = importlib.import_module(f"{package_name}.{name}")
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return module attributes including lazily exposed subpackages."""
    return sorted({*globals().keys(), *_SUBPACKAGES})
