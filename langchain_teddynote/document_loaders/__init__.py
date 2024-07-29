import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hwp import HWPReader
    from pdf import PDFParser


_module_lookup = {
    "HWPLoader": "document_loaders.hwp",
    "PDFParser": "document_loaders.pdf",
}

from .hwp import HWPLoader
from .pdf import PDFParser


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "HWPLoader",
    "PDFParser",
]
