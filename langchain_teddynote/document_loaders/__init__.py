import importlib
from typing import Any


_module_lookup = {
    "HWPLoader": "document_loaders.hwp"
}

from .hwp import HWPLoader


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "HWPLoader",
]
