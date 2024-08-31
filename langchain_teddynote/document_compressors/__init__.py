import importlib
from typing import Any

_module_lookup = {
    "LLMChainExtractor": "document_compressors.llm_extractor",
    "LLMChainFilter": "document_compressors.llm_filter",
}

from .llm_extractor import LLMChainExtractor
from .llm_filter import LLMChainFilter


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "LLMChainExtractor",
    "LLMChainFilter",
]
