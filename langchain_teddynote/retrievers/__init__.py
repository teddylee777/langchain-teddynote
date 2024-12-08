import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from retrievers.kiwi_bm25 import KiwiBM25Retriever

_module_lookup = {
    "KiwiBM25Retriever": "retrievers.kiwi_bm25",
    "EnsembleRetriever": "retrievers.ensemble",
    "EnsembleMethod": "retrievers.ensemble",
}

from .kiwi_bm25 import KiwiBM25Retriever
from .ensemble import EnsembleRetriever, EnsembleMethod


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "KiwiBM25Retriever",
    "EnsembleRetriever",
    "EnsembleMethod",
]
