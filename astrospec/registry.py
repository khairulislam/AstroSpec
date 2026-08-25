"""A small timm-style model registry: register_model, create_model, list_models, is_model."""

from typing import Callable, Dict

_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(fn: Callable) -> Callable:
    """Decorator that registers a factory function under its own name."""
    _MODEL_REGISTRY[fn.__name__] = fn
    return fn


def create_model(name: str, **kwargs):
    """Construct a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_models()}")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models():
    return sorted(_MODEL_REGISTRY.keys())


def is_model(name: str) -> bool:
    return name in _MODEL_REGISTRY
