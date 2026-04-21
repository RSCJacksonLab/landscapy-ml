from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    runtime_checkable,
)

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from fitness_landscape.core.fitness import NumericFitness, ProbabilisticCategoricalFitness
except Exception:  # pragma: no cover - optional dependency
    NumericFitness = Any  # type: ignore
    ProbabilisticCategoricalFitness = Any  # type: ignore


DEFAULT_EMBEDDING_MODEL = "facebook/esm2_t6_8M_UR50D"


def resolve_embedding_info(
    landscape: Any,
) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
    domain = getattr(
        landscape,
        "active_embedding_domain",
        getattr(landscape, "_active_embedding_domain", None),
    )
    meta = None
    if hasattr(landscape, "get_embedding_metadata"):
        try:
            meta = landscape.get_embedding_metadata(domain)
        except Exception:
            meta = None
    model_name = getattr(landscape, "embedding_model", None)
    if meta is not None and meta.get("model_name"):
        model_name = meta.get("model_name")
    return domain, model_name, meta


@runtime_checkable
class ModelAdapter(Protocol):
    layer_kind: str

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor] | torch.Tensor:
        ...


class DefaultModelAdapter:
    def __init__(self, model: Any, layer_kind: str) -> None:
        self.model = model
        self.layer_kind = layer_kind
        self.embedding_domain = getattr(model, "embedding_domain", None)
        self.embedding_model = getattr(model, "embedding_model", None)

    def eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

    def to(self, device: torch.device | str) -> "DefaultModelAdapter":
        if hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor] | torch.Tensor:
        if self.layer_kind == "prob_categorical":
            if hasattr(self.model, "predict_with_uncertainty"):
                mean, var = self.model.predict_with_uncertainty(inputs)
                return {"mean": mean, "var": var}
            if hasattr(self.model, "predict"):
                out = self.model.predict(inputs)
                if isinstance(out, Mapping):
                    return out
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    return {"mean": out[0], "var": out[1]}
                return {"mean": out}

        if hasattr(self.model, "predict"):
            out = self.model.predict(inputs)
        else:
            out = self.model(inputs)
        if isinstance(out, Mapping):
            return out
        return {"output": out}


ModelAdapterFactory = Callable[[Any], ModelAdapter]
_MODEL_ADAPTERS: dict[Type[Any], ModelAdapterFactory] = {}

_MODEL_TO_LAYER: dict[Type[Any], str] = {}


def register_model_adapter(
    model_cls: Type[Any], adapter_factory: ModelAdapterFactory, *, overwrite: bool = False
) -> None:
    if model_cls in _MODEL_ADAPTERS and not overwrite:
        raise ValueError(
            f"Adapter for model class {model_cls.__name__} already registered."
        )
    _MODEL_ADAPTERS[model_cls] = adapter_factory


def register_model_layer_mapping(
    model_cls: Type[Any], layer_kind: str, *, overwrite: bool = False
) -> None:
    if model_cls in _MODEL_TO_LAYER and not overwrite:
        raise ValueError(
            f"Model {model_cls.__name__} already mapped to {_MODEL_TO_LAYER[model_cls]!r}."
        )
    _MODEL_TO_LAYER[model_cls] = layer_kind


def resolve_model_adapter(model: Any) -> ModelAdapter:
    if isinstance(model, ModelAdapter):
        return model
    model_type = type(model)
    for cls in model_type.mro():
        factory = _MODEL_ADAPTERS.get(cls)
        if factory is not None:
            return factory(model)
    layer_kind = getattr(model, "layer_kind", None) or _MODEL_TO_LAYER.get(model_type)
    if layer_kind is None:
        raise ValueError(
            f"Model type {model_type.__name__} is not supported for inference."
        )
    return DefaultModelAdapter(model, layer_kind)


def infer_device(model: Any) -> Optional[torch.device]:
    if model is None:
        return None
    device = getattr(model, "device", None)
    if device is not None:
        try:
            return torch.device(device)
        except Exception:
            pass
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return None
    return None


def normalize_adapter_outputs(
    outputs: Any, layer_kind: str
) -> Mapping[str, torch.Tensor]:
    if isinstance(outputs, Mapping):
        return outputs
    if torch.is_tensor(outputs):
        key = "mean" if layer_kind == "prob_categorical" else "output"
        return {key: outputs}
    if isinstance(outputs, (tuple, list)):
        if layer_kind == "prob_categorical" and len(outputs) == 2:
            return {"mean": outputs[0], "var": outputs[1]}
        if len(outputs) == 1 and torch.is_tensor(outputs[0]):
            return {"output": outputs[0]}
    raise ValueError("Model adapter returned unsupported outputs.")


class LandscapeInputAdapter(ABC):
    name: str = "input_adapter"

    def metadata(self, landscape: Any) -> Mapping[str, Any]:
        return {"input_adapter": self.name}

    def embedding_info(
        self, landscape: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
        return None, None, None

    @abstractmethod
    def iter_batches(
        self,
        landscape: Any,
        *,
        batch_size: int,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        ...

    @abstractmethod
    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        ...


class EmbeddingInputAdapter(LandscapeInputAdapter):
    name = "embedding"

    def __init__(self, *, default_model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.default_model_name = default_model_name

    def embedding_info(
        self, landscape: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
        return resolve_embedding_info(landscape)

    def metadata(self, landscape: Any) -> Mapping[str, Any]:
        domain, model_name, meta = self.embedding_info(landscape)
        info: dict[str, Any] = {"input_adapter": self.name}
        if domain is not None:
            info["embedding_domain"] = domain
        if model_name is not None:
            info["embedding_model"] = model_name
        if isinstance(meta, Mapping) and "embedding_mode" in meta:
            info["embedding_mode"] = meta["embedding_mode"]
        return info

    def iter_batches(
        self,
        landscape: Any,
        *,
        batch_size: int,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        emb_array = landscape.get_embedding()
        if emb_array is None:
            fallback_model = None
            _, active_model, active_meta = resolve_embedding_info(landscape)
            if isinstance(active_meta, Mapping):
                fallback_model = active_meta.get("model_name")
            model_name = (
                model_name or active_model or fallback_model or self.default_model_name
            )
            if not hasattr(landscape, "compute_plm_embeddings"):
                raise RuntimeError(
                    "No embeddings available on landscape and it cannot compute embeddings."
                )
            landscape.compute_plm_embeddings(model_name=model_name)
            emb_array = landscape.get_embedding()
        if emb_array is None:
            raise RuntimeError(
                "No embeddings available on landscape and automatic computation failed."
            )
        emb = torch.as_tensor(emb_array, dtype=torch.float32)
        ds = TensorDataset(emb)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        for batch in dl:
            yield batch

    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        if device is not None and torch.is_tensor(inputs):
            return inputs.to(device)
        return inputs


class GraphTensorInputAdapter(LandscapeInputAdapter):
    """
    Generic adapter for models that consume ``landscape.to_graph_tensor()``.
    """

    name = "graph_tensor"

    def __init__(self, *, tokenizer: Any | str | None = None) -> None:
        self.tokenizer = tokenizer

    def metadata(self, landscape: Any) -> Mapping[str, Any]:  # noqa: ARG002
        info = {"input_adapter": self.name, "graph_tensor": True}
        if self.tokenizer is not None:
            info["tokenizer"] = str(self.tokenizer)
        return info

    def iter_batches(
        self,
        landscape: Any,
        *,
        batch_size: int,  # noqa: ARG002 - graph models operate on the full graph
        num_workers: int = 0,  # noqa: ARG002
        device: Optional[torch.device] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterable[Any]:
        if not hasattr(landscape, "to_graph_tensor"):
            raise RuntimeError("Landscape does not implement to_graph_tensor().")
        yield landscape.to_graph_tensor(tokenizer=self.tokenizer)

    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        if device is not None and hasattr(batch, "to"):
            return batch.to(device)
        return batch


class NodeIndexInputAdapter(LandscapeInputAdapter):
    """
    Generic adapter that exposes a landscape as batches of node indices.
    """

    name = "node_index"

    def metadata(self, landscape: Any) -> Mapping[str, Any]:  # noqa: ARG002
        return {
            "input_adapter": self.name,
            "graph_node_index": True,
        }

    def iter_batches(
        self,
        landscape: Any,
        *,
        batch_size: int,
        num_workers: int = 0,
        device: Optional[torch.device] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterable[Any]:
        if hasattr(landscape, "sequences"):
            num_items = len(landscape.sequences)
        else:
            try:
                num_items = len(landscape)
            except TypeError as exc:
                raise RuntimeError(
                    "Landscape does not expose a sequence count for node-index batching."
                ) from exc
        indices = torch.arange(num_items, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(indices)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for (batch_indices,) in loader:
            yield batch_indices

    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        if device is not None and torch.is_tensor(tensor):
            return tensor.to(device)
        return tensor


InputAdapterFactory = Callable[..., LandscapeInputAdapter]
_INPUT_ADAPTERS: dict[str, InputAdapterFactory] = {}


def register_input_adapter(
    name: str, factory: InputAdapterFactory, *, overwrite: bool = False
) -> None:
    if name in _INPUT_ADAPTERS and not overwrite:
        raise ValueError(f"Input adapter '{name}' is already registered.")
    _INPUT_ADAPTERS[name] = factory


def resolve_input_adapter(
    adapter: LandscapeInputAdapter | str | None, **kwargs: Any
) -> LandscapeInputAdapter:
    if isinstance(adapter, LandscapeInputAdapter):
        return adapter
    name = adapter or "embedding"
    if name not in _INPUT_ADAPTERS:
        available = ", ".join(sorted(_INPUT_ADAPTERS)) or "none"
        raise ValueError(f"Unknown input adapter '{name}'. Available: {available}")
    return _INPUT_ADAPTERS[name](**kwargs)


register_input_adapter("embedding", EmbeddingInputAdapter, overwrite=True)
register_input_adapter("graph_tensor", GraphTensorInputAdapter, overwrite=True)
register_input_adapter("node_index", NodeIndexInputAdapter, overwrite=True)


class LandscapeOutputAdapter(ABC):
    layer_kind: str

    @abstractmethod
    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        ...


class FunctionOutputAdapter(LandscapeOutputAdapter):
    def __init__(self, layer_kind: str, func: Callable[..., Any]) -> None:
        self.layer_kind = layer_kind
        self._func = func

    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        return self._func(outputs, categories, metadata, layer_name)


class ProbCategoricalOutputAdapter(LandscapeOutputAdapter):
    layer_kind = "prob_categorical"

    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        if "mean" not in outputs:
            raise ValueError(
                "Probabilistic categorical adapter requires 'mean' prediction."
            )
        mean = outputs["mean"]
        var = outputs.get("var")
        num_classes = mean.shape[-1]
        cats = (
            list(categories)
            if categories is not None
            else [f"class_{i}" for i in range(num_classes)]
        )
        if len(cats) != num_classes:
            raise ValueError(
                "Number of categories does not match model output dimension."
            )
        meta = dict(metadata)
        if var is not None:
            meta["variance"] = var.numpy()
        return ProbabilisticCategoricalFitness(
            name=layer_name,
            probabilities=mean.numpy(),
            categories=cats,
            metadata=meta,
        )


class NumericOutputAdapter(LandscapeOutputAdapter):
    layer_kind = "numeric"

    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],  # noqa: ARG002 - not used for numeric layers
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        tensor = outputs.get("output")
        if tensor is None and len(outputs) == 1:
            tensor = next(iter(outputs.values()))
        if tensor is None:
            raise ValueError(
                "Numeric adapter expects a single tensor output or an 'output' key."
            )
        if not torch.is_tensor(tensor):
            raise TypeError("Numeric adapter received non-tensor output.")

        arr = tensor.detach().cpu()
        if arr.ndim == 0:
            arr = arr.view(1, 1)
        elif arr.ndim == 1:
            arr = arr.unsqueeze(-1)
        elif arr.ndim > 2:
            raise ValueError("Numeric adapter expects a 1-D or 2-D tensor output.")

        meta = dict(metadata)
        if hasattr(NumericFitness, "from_tensor"):
            return NumericFitness.from_tensor(
                name=layer_name,
                tensor=arr,
                metadata=meta,
            )
        return NumericFitness(
            name=layer_name,
            values=arr.numpy().tolist(),
            metadata=meta,
        )


OutputAdapterFactory = Callable[[], LandscapeOutputAdapter]
_OUTPUT_ADAPTERS: dict[str, OutputAdapterFactory] = {}


def register_output_adapter(
    kind: str,
    adapter: LandscapeOutputAdapter | type[LandscapeOutputAdapter],
    *,
    overwrite: bool = False,
) -> None:
    if kind in _OUTPUT_ADAPTERS and not overwrite:
        raise ValueError(f"Output adapter for layer kind {kind!r} already exists.")
    if isinstance(adapter, LandscapeOutputAdapter):
        _OUTPUT_ADAPTERS[kind] = lambda adapter=adapter: adapter
        return
    if isinstance(adapter, type) and issubclass(adapter, LandscapeOutputAdapter):
        _OUTPUT_ADAPTERS[kind] = adapter
        return
    raise TypeError("Output adapter must be an adapter instance or subclass.")


def register_layer_adapter(
    kind: str, adapter: Callable[..., Any], *, overwrite: bool = False
) -> None:
    output_adapter = FunctionOutputAdapter(kind, adapter)
    register_output_adapter(kind, output_adapter, overwrite=overwrite)


def resolve_output_adapter(kind: str) -> LandscapeOutputAdapter:
    if kind not in _OUTPUT_ADAPTERS:
        raise ValueError(f"No adapter registered for layer kind '{kind}'.")
    return _OUTPUT_ADAPTERS[kind]()


register_output_adapter(
    NumericOutputAdapter.layer_kind, NumericOutputAdapter, overwrite=True
)
register_output_adapter(
    ProbCategoricalOutputAdapter.layer_kind, ProbCategoricalOutputAdapter, overwrite=True
)
