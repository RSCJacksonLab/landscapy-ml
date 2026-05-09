from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl

ModelFactory = Callable[..., pl.LightningModule]
DataFactory = Callable[..., pl.LightningDataModule]

_SPLIT_INDEX_ALIASES: dict[str, str] = {
    "train": "train_indices",
    "training": "train_indices",
    "train_indices": "train_indices",
    "val": "val_indices",
    "valid": "val_indices",
    "validation": "val_indices",
    "val_indices": "val_indices",
    "validation_indices": "val_indices",
    "test": "test_indices",
    "test_indices": "test_indices",
}


@dataclass(frozen=True)
class ModelRegistryEntry:
    factory: ModelFactory
    requires_num_features: bool = True


_MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {}
_DATA_REGISTRY: Dict[str, DataFactory] = {}
_BUILTINS_REGISTERED = False


def normalize_split_indices(
    split_indices: Mapping[str, Sequence[int]] | None,
) -> dict[str, list[int]]:
    """
    Normalize common split names to data-builder keyword arguments.
    """

    if split_indices is None:
        return {}

    normalized: dict[str, list[int]] = {}
    for raw_name, values in split_indices.items():
        key = _SPLIT_INDEX_ALIASES.get(str(raw_name).lower())
        if key is None:
            valid = ", ".join(sorted(_SPLIT_INDEX_ALIASES))
            raise ValueError(
                f"Unknown split name {raw_name!r}. Expected one of: {valid}."
            )
        if values is None:
            continue
        if key in normalized:
            raise ValueError(f"Split indices for {key!r} were supplied more than once.")
        normalized[key] = [int(idx) for idx in values]
    return normalized


def factory_accepts_kwargs(factory: Callable[..., Any], names: Sequence[str]) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True
    return all(name in params for name in names)


def register_model(
    name: str,
    factory: ModelFactory,
    *,
    overwrite: bool = False,
    requires_num_features: bool = True,
) -> None:
    if name in _MODEL_REGISTRY and not overwrite:
        raise ValueError(f"Model '{name}' is already registered.")
    _MODEL_REGISTRY[name] = ModelRegistryEntry(
        factory=factory, requires_num_features=requires_num_features
    )


def register_data(name: str, factory: DataFactory, *, overwrite: bool = False) -> None:
    if name in _DATA_REGISTRY and not overwrite:
        raise ValueError(f"Data builder '{name}' is already registered.")
    _DATA_REGISTRY[name] = factory


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def available_data_builders() -> list[str]:
    return sorted(_DATA_REGISTRY)


def get_model_entry(name: str) -> ModelRegistryEntry:
    return _MODEL_REGISTRY[name]


def get_data_factory(name: str) -> DataFactory:
    return _DATA_REGISTRY[name]


def _load_object(path: str) -> Any:
    if "." not in path:
        raise ValueError(
            "class_path must be a fully qualified module path, e.g. mypkg.models.MyModel"
        )
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Could not find '{attr}' in module '{module_path}'.") from exc


def build_external_model(
    *,
    class_path: str,
    init_args: Optional[Sequence[Any]] = None,
    init_kwargs: Optional[dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    adapter_args: Optional[Sequence[Any]] = None,
    adapter_kwargs: Optional[dict[str, Any]] = None,
    **extra_kwargs: Any,
) -> pl.LightningModule:
    """
    Instantiate a LightningModule from a class path, with optional wrapping.
    """

    target = _load_object(class_path)
    args = list(init_args) if init_args is not None else []
    kwargs = dict(init_kwargs or {})
    kwargs.update(extra_kwargs)
    model = target(*args, **kwargs)

    if adapter_path:
        adapter = _load_object(adapter_path)
        adapter_args = list(adapter_args) if adapter_args is not None else []
        adapter_kwargs = dict(adapter_kwargs or {})
        model = adapter(model, *adapter_args, **adapter_kwargs)

    if not isinstance(model, pl.LightningModule):
        raise TypeError(
            "External model must be a LightningModule. "
            "Provide an adapter that wraps the model into a LightningModule."
        )
    return model


def register_builtin_components() -> None:
    """
    Register the small built-in models and landscapy data builders.

    The registry remains functional and explicit: custom projects can import
    this module, register their own factories, and use those names through the
    CLI or ``TrainingJob``.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from .data import (
        LandscapeDataModule,
        LandscapeGraphRegressionDataModule,
    )

    register_model(
        "external",
        build_external_model,
        overwrite=True,
        requires_num_features=False,
    )

    register_data("landscape_records", LandscapeDataModule, overwrite=True)
    register_data(
        "landscape_graph_regression",
        LandscapeGraphRegressionDataModule.from_landscape,
        overwrite=True,
    )

    _BUILTINS_REGISTERED = True


register_builtin_components()


__all__ = [
    "DataFactory",
    "ModelFactory",
    "ModelRegistryEntry",
    "_DATA_REGISTRY",
    "_MODEL_REGISTRY",
    "available_data_builders",
    "available_models",
    "build_external_model",
    "factory_accepts_kwargs",
    "get_data_factory",
    "get_model_entry",
    "normalize_split_indices",
    "register_builtin_components",
    "register_data",
    "register_model",
]
