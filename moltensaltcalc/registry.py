from typing import Callable

MODEL_REGISTRY = {}
MODEL_METADATA = {}


def register_model(name: str, metadata: dict | None = None) -> Callable:
    """Loads a uMLIP model and registers it with the simulator.

    Args:
        name (str): Name of the model to register.
        metadata (dict | None, optional): Metadata for the model. Defaults to None.

    Raises:
        ValueError: If the model is already registered.

    Returns:
        Callable: Decorator function that checks and logs the model registration.
    """
    name = name.lower()
    metadata = metadata or {}

    def decorator(fn):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")

        MODEL_REGISTRY[name] = fn
        MODEL_METADATA[name] = metadata

        return fn

    return decorator
