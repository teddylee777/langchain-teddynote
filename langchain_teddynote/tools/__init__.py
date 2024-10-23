import importlib
from typing import Any

_module_lookup = {
    "GoogleNews": "tools.news",
    "TavilySearch": "tools.tavily",
}

from .news import GoogleNews
from .tavily import TavilySearch


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "GoogleNews",
    "TavilySearch",
]
