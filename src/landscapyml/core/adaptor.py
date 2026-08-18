from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

from ._optional import is_missing_optional_dependency
from .data_utils import LandscapeRecord

try:
    from fitness_landscape.core.fitness import (
        NumericFitness,
        ProbabilisticCategoricalFitness,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if not is_missing_optional_dependency(exc, "fitness_landscape"):
        raise
    NumericFitness = None  # type: ignore
    ProbabilisticCategoricalFitness = None  # type: ignore
    _LANDSCAPY_FITNESS_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _LANDSCAPY_FITNESS_IMPORT_ERROR = None


DEFAULT_EMBEDDING_MODEL = "facebook/esm2_t6_8M_UR50D"


def resolve_embedding_info(
    landscape: Any,
) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
    domain = getattr(
        landscape,
        "active_embedding_domain",
        getattr(landscape, "_active_embedding_domain", None),
    )
    meta = (
        landscape.get_embedding_metadata(domain)
        if hasattr(landscape, "get_embedding_metadata")
        else None
    )
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


def _extract_label_mapping(layer: Any) -> Optional[list[str]]:
    cats = getattr(layer, "categories", None)
    if cats is None:
        meta = getattr(layer, "metadata", None)
        if isinstance(meta, Mapping):
            cats = meta.get("categories")
    if cats is None:
        return None
    return list(cats)


@dataclass(frozen=True)
class LandscapeExport:
    records: list[LandscapeRecord]
    fitness_mappings: dict[str, Optional[list[str]]]


def _copy_record_views(record: Mapping[str, Any]) -> LandscapeRecord:
    copied: LandscapeRecord = {
        "sequence_tensor": record.get("sequence_tensor"),
        "fitness_tensors": dict(record.get("fitness_tensors") or {}),
    }
    if "embedding" in record:
        copied["embedding"] = record["embedding"]
    if "attention_mask" in record:
        copied["attention_mask"] = record["attention_mask"]
    return copied


def export_landscape_records(
    landscape: Any,
    *,
    fitness_layers: Optional[Sequence[str]] = None,
    rename_fitness: Optional[Mapping[str, str]] = None,
    feature_view: str = "auto",
    include_embeddings: bool = True,
    tokenizer: Any | str | None = None,
    sequence_idx: Optional[Sequence[int]] = None,
    sequence: Optional[Sequence[str] | str] = None,
) -> LandscapeExport:
    """
    Export a ``FitnessLandscape`` into task-agnostic ML record dictionaries.
    """

    if not hasattr(landscape, "to_sequence_tensors"):
        raise ValueError("Landscape must implement to_sequence_tensors.")

    raw = landscape.to_sequence_tensors(
        sequence_idx=sequence_idx,
        sequence=sequence,
        tokenizer=tokenizer,
        feature_view=feature_view,
        include_embeddings=include_embeddings,
        as_batch=False,
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("to_sequence_tensors must return a sequence of record mappings.")

    requested_layers = list(fitness_layers) if fitness_layers is not None else None
    rename_map = dict(rename_fitness or {})

    records: list[LandscapeRecord] = []
    selected_layer_names: list[str] | None = requested_layers

    for rec in raw:
        if not isinstance(rec, Mapping):
            raise ValueError("Landscape export must yield mapping records.")

        fitness = rec.get("fitness_tensors")
        if not isinstance(fitness, Mapping):
            raise ValueError("Landscape export records must contain a fitness_tensors mapping.")

        if selected_layer_names is None:
            selected_layer_names = list(fitness.keys())

        missing = [name for name in selected_layer_names if name not in fitness]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Requested fitness layer(s) missing from landscape export: {missing_list}."
            )

        new_rec = _copy_record_views(rec)
        new_rec["fitness_tensors"] = {
            rename_map.get(name, name): fitness[name] for name in selected_layer_names
        }
        records.append(new_rec)

    if selected_layer_names is None:
        selected_layer_names = []

    available_layers = getattr(landscape, "fitness_layers", None)
    fitness_mappings: dict[str, Optional[list[str]]] = {}
    for layer_name in selected_layer_names:
        exported_name = rename_map.get(layer_name, layer_name)
        mapping = None
        if isinstance(available_layers, Mapping):
            mapping = _extract_label_mapping(available_layers.get(layer_name))
        fitness_mappings[exported_name] = mapping

    return LandscapeExport(records=records, fitness_mappings=fitness_mappings)


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
        if hasattr(landscape, "to_graph_tensor"):
            try:
                yield landscape.to_graph_tensor(tokenizer=self.tokenizer)
                return
            except ValueError as exc:
                if "inhomogeneous shape" not in str(exc):
                    raise
        if not hasattr(landscape, "graph"):
            raise RuntimeError("Landscape does not implement to_graph_tensor().")

        from .data import _graph_tensor_from_landscape_graph

        yield _graph_tensor_from_landscape_graph(landscape)

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
        if ProbabilisticCategoricalFitness is None:
            raise ImportError(
                "Probabilistic fitness output requires landscapy to be installed."
            ) from _LANDSCAPY_FITNESS_IMPORT_ERROR
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
        if NumericFitness is None:
            raise ImportError(
                "Numeric fitness output requires landscapy to be installed."
            ) from _LANDSCAPY_FITNESS_IMPORT_ERROR
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
