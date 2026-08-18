"""Registries and adapters connecting Landscapy objects to PyTorch models."""

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
    """Read active embedding provenance from a landscape.

    Parameters
    ----------
    landscape : Any
        Object exposing Landscapy embedding attributes or metadata accessors.

    Returns
    -------
    domain : str or None
        Active embedding domain.
    model_name : str or None
        Embedding model identifier from metadata or the legacy attribute.
    metadata : mapping or None
        Active-domain embedding metadata. The mapping is not copied.
    """
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
    """Define the minimal inference interface consumed by landscapy-ml.

    Attributes
    ----------
    layer_kind : str
        Logical output kind resolved through the output-adapter registry.
    """

    layer_kind: str

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor] | torch.Tensor:
        """Return model predictions for one input batch.

        Parameters
        ----------
        inputs : Any
            Model-specific batch produced by a landscape input adapter.

        Returns
        -------
        mapping of str to torch.Tensor or torch.Tensor
            Named or unnamed prediction tensors. Tensors may reside on the
            model device and may require gradients.
        """
        ...


class DefaultModelAdapter:
    """Normalize conventional model prediction methods.

    Parameters
    ----------
    model : Any
        Callable model or object implementing ``predict`` or
        ``predict_with_uncertainty``.
    layer_kind : str
        Logical output kind used to normalize return keys.

    Attributes
    ----------
    model : Any
        Wrapped model.
    layer_kind : str
        Logical output kind.
    embedding_domain : str or None
        Optional embedding domain declared by the model.
    embedding_model : str or None
        Optional embedding model identifier declared by the model.
    """

    def __init__(self, model: Any, layer_kind: str) -> None:
        self.model = model
        self.layer_kind = layer_kind
        self.embedding_domain = getattr(model, "embedding_domain", None)
        self.embedding_model = getattr(model, "embedding_model", None)

    def eval(self) -> None:
        """Place the wrapped model in evaluation mode when supported."""
        if hasattr(self.model, "eval"):
            self.model.eval()

    def to(self, device: torch.device | str) -> "DefaultModelAdapter":
        """Move the wrapped model to a device when supported.

        Parameters
        ----------
        device : torch.device or str
            Target PyTorch device.

        Returns
        -------
        DefaultModelAdapter
            This adapter for method chaining.
        """
        if hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor] | torch.Tensor:
        """Run the wrapped model and normalize conventional output keys.

        Parameters
        ----------
        inputs : Any
            Model-specific input batch.

        Returns
        -------
        mapping of str to torch.Tensor or torch.Tensor
            ``mean`` and optional ``var`` for probabilistic categorical
            outputs, or ``output`` for ordinary predictions. Existing mappings
            pass through unchanged.
        """
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
    """Register a model-adapter factory for a model class.

    Parameters
    ----------
    model_cls : type
        Model class used as the registry key.
    adapter_factory : callable
        Factory accepting a model instance and returning a ``ModelAdapter``.
    overwrite : bool, default=False
        Replace an existing exact-class registration.

    Returns
    -------
    None
        The process-local registry is mutated.

    Raises
    ------
    ValueError
        If ``model_cls`` is registered and ``overwrite`` is false.

    Notes
    -----
    Resolution follows Python method resolution order. Adapter factories take
    precedence over model-to-layer mappings.
    """
    if model_cls in _MODEL_ADAPTERS and not overwrite:
        raise ValueError(
            f"Adapter for model class {model_cls.__name__} already registered."
        )
    _MODEL_ADAPTERS[model_cls] = adapter_factory


def register_model_layer_mapping(
    model_cls: Type[Any], layer_kind: str, *, overwrite: bool = False
) -> None:
    """Map a model class to a logical output-layer kind.

    Parameters
    ----------
    model_cls : type
        Model class used as the registry key.
    layer_kind : str
        Key resolved through the output-adapter registry.
    overwrite : bool, default=False
        Replace an existing exact-class mapping.

    Returns
    -------
    None
        The process-local registry is mutated.

    Raises
    ------
    ValueError
        If ``model_cls`` is mapped and ``overwrite`` is false.

    Notes
    -----
    Subclasses inherit mappings according to Python method resolution order;
    an exact subclass mapping has priority over base-class mappings.
    """
    if model_cls in _MODEL_TO_LAYER and not overwrite:
        raise ValueError(
            f"Model {model_cls.__name__} already mapped to {_MODEL_TO_LAYER[model_cls]!r}."
        )
    _MODEL_TO_LAYER[model_cls] = layer_kind


def resolve_model_adapter(model: Any) -> ModelAdapter:
    """Resolve a model instance to the inference adapter protocol.

    Parameters
    ----------
    model : Any
        Model adapter, registered model instance, or model declaring
        ``layer_kind``.

    Returns
    -------
    ModelAdapter
        Supplied adapter, registered custom adapter, or default wrapper.

    Raises
    ------
    ValueError
        If no adapter or output-layer mapping supports the model class.
    """
    if isinstance(model, ModelAdapter):
        return model
    model_type = type(model)
    for cls in model_type.mro():
        factory = _MODEL_ADAPTERS.get(cls)
        if factory is not None:
            return factory(model)
    layer_kind = getattr(model, "layer_kind", None)
    if not layer_kind:
        for cls in model_type.mro():
            layer_kind = _MODEL_TO_LAYER.get(cls)
            if layer_kind is not None:
                break
    if layer_kind is None:
        raise ValueError(
            f"Model type {model_type.__name__} is not supported for inference."
        )
    return DefaultModelAdapter(model, layer_kind)


def infer_device(model: Any) -> Optional[torch.device]:
    """Infer a PyTorch device from a model-like object.

    Parameters
    ----------
    model : Any
        Object with an optional ``device`` attribute or parameter iterator.

    Returns
    -------
    torch.device or None
        Explicit device, first-parameter device, or ``None`` for an absent or
        parameterless model.
    """
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
    """Normalize adapter outputs to a named tensor mapping.

    Parameters
    ----------
    outputs : Any
        Existing mapping, tensor, or supported tuple/list output.
    layer_kind : str
        Logical layer kind. ``prob_categorical`` tensors use the ``mean`` key
        and two-element sequences use ``mean`` and ``var``.

    Returns
    -------
    mapping of str to torch.Tensor
        Normalized output mapping without tensor copies or device transfers.

    Raises
    ------
    ValueError
        If ``outputs`` has no supported representation.
    """
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
    """Store task-agnostic landscape records and categorical labels.

    Parameters
    ----------
    records : list of dict
        Per-sequence ML records aligned with landscape sequence order.
    fitness_mappings : dict of str to list of str or None
        Category order for each exported fitness field, or ``None`` for
        non-categorical fields.

    Attributes
    ----------
    records : list of dict
        Exported records.
    fitness_mappings : dict of str to list of str or None
        Exported categorical label mappings.
    """

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
    """Export a fitness landscape into task-agnostic ML records.

    Parameters
    ----------
    landscape : Any
        Landscapy object implementing ``to_sequence_tensors``.
    fitness_layers : sequence of str or None, optional
        Fitness layers to include. ``None`` selects every layer returned by
        the first record.
    rename_fitness : mapping of str to str or None, optional
        Source-to-exported fitness key mapping.
    feature_view : str, default="auto"
        Feature representation forwarded to Landscapy.
    include_embeddings : bool, default=True
        Include active embeddings in exported records when available.
    tokenizer : Any, str, or None, optional
        Tokenizer forwarded to Landscapy tensor export.
    sequence_idx : sequence of int or None, optional
        Canonical sequence positions to export.
    sequence : sequence of str, str, or None, optional
        Sequence identifiers or values to export.

    Returns
    -------
    LandscapeExport
        Records plus category-order metadata for selected fitness layers.

    Raises
    ------
    ValueError
        If the landscape lacks tensor export, returns malformed records, or
        omits a requested fitness layer.

    Notes
    -----
    Record tensors are retained by reference; the top-level records and their
    ``fitness_tensors`` mappings are copied.
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
    """Define extraction and device transfer for landscape model inputs.

    Attributes
    ----------
    name : str
        Registry key and metadata identifier for the adapter.
    """

    name: str = "input_adapter"

    def metadata(self, landscape: Any) -> Mapping[str, Any]:
        """Describe the input representation used for inference.

        Parameters
        ----------
        landscape : Any
            Source landscape.

        Returns
        -------
        mapping of str to Any
            Metadata suitable for attaching to the predicted fitness layer.
        """
        return {"input_adapter": self.name}

    def embedding_info(
        self, landscape: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
        """Return embedding provenance required by the adapter.

        Parameters
        ----------
        landscape : Any
            Source landscape.

        Returns
        -------
        domain : str or None
            Active embedding domain, if relevant.
        model_name : str or None
            Embedding model identifier, if relevant.
        metadata : mapping or None
            Embedding provenance mapping, if relevant.
        """
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
        """Yield model-specific batches from a landscape.

        Parameters
        ----------
        landscape : Any
            Source landscape.
        batch_size : int
            Requested number of records per batch.
        num_workers : int, default=0
            Worker-process count for adapters backed by DataLoader.
        device : torch.device or None, optional
            Intended model device.
        **kwargs : Any
            Adapter-specific extraction options.

        Yields
        ------
        Any
            Raw adapter-specific batches.
        """
        ...

    @abstractmethod
    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        """Convert a raw adapter batch into model inputs.

        Parameters
        ----------
        batch : Any
            Raw batch yielded by :meth:`iter_batches`.
        device : torch.device or None, optional
            Target model device.

        Returns
        -------
        Any
            Model-ready input object.
        """
        ...


class EmbeddingInputAdapter(LandscapeInputAdapter):
    """Batch a landscape's active floating-point embedding matrix.

    Parameters
    ----------
    default_model_name : str, default=DEFAULT_EMBEDDING_MODEL
        Model used for automatic PLM embedding when the landscape has no
        cached active embedding.

    Attributes
    ----------
    default_model_name : str
        Automatic embedding model identifier.
    """

    name = "embedding"

    def __init__(self, *, default_model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.default_model_name = default_model_name

    def embedding_info(
        self, landscape: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[Mapping[str, Any]]]:
        """Return active embedding domain, model, and metadata.

        Parameters
        ----------
        landscape : Any
            Source landscape.

        Returns
        -------
        domain : str or None
            Active embedding domain.
        model_name : str or None
            Active embedding model identifier.
        metadata : mapping or None
            Active embedding provenance.
        """
        return resolve_embedding_info(landscape)

    def metadata(self, landscape: Any) -> Mapping[str, Any]:
        """Build predicted-layer metadata for the active embeddings.

        Parameters
        ----------
        landscape : Any
            Source landscape.

        Returns
        -------
        mapping of str to Any
            Adapter name and available embedding domain, model, and mode.
        """
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
        """Yield ``float32`` embedding batches on their source device.

        Parameters
        ----------
        landscape : Any
            Landscape implementing ``get_embedding`` and optionally
            ``compute_plm_embeddings``.
        batch_size : int
            Number of embedding rows per batch.
        num_workers : int, default=0
            DataLoader worker-process count.
        device : torch.device or None, optional
            Accepted for the common interface; transfer occurs in
            :meth:`to_model_inputs`.
        model_name : str or None, optional
            PLM identifier used only for automatic embedding.
        **kwargs : Any
            Ignored compatibility options.

        Yields
        ------
        tuple of torch.Tensor
            Single-item DataLoader batches with shape
            ``(batch, embedding_dim)`` and dtype ``float32``.

        Raises
        ------
        RuntimeError
            If embeddings are unavailable and cannot be computed.

        Notes
        -----
        Automatic embedding mutates the source landscape by caching the
        computed PLM embedding.
        """
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
        """Extract an embedding tensor and optionally move it to a device.

        Parameters
        ----------
        batch : Any
            DataLoader tuple/list or tensor batch.
        device : torch.device or None, optional
            Target device. ``None`` preserves the current device.

        Returns
        -------
        Any
            Tensor input when recognized, otherwise the batch value unchanged.
        """
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        if device is not None and torch.is_tensor(inputs):
            return inputs.to(device)
        return inputs


class GraphTensorInputAdapter(LandscapeInputAdapter):
    """Adapt a complete landscape graph to a graph-native model.

    Parameters
    ----------
    tokenizer : Any, str, or None, optional
        Tokenizer forwarded to ``landscape.to_graph_tensor``.

    Attributes
    ----------
    tokenizer : Any, str, or None
        Configured graph-export tokenizer.

    Notes
    -----
    The full graph is yielded once; ``batch_size`` does not partition nodes.
    Variable-length tensor-export failures fall back to graph edges plus active
    embeddings or sequence-composition features.
    """

    name = "graph_tensor"

    def __init__(self, *, tokenizer: Any | str | None = None) -> None:
        self.tokenizer = tokenizer

    def metadata(self, landscape: Any) -> Mapping[str, Any]:  # noqa: ARG002
        """Describe graph-tensor input provenance.

        Parameters
        ----------
        landscape : Any
            Source landscape; accepted for the common adapter interface.

        Returns
        -------
        mapping of str to Any
            Graph-input marker and optional tokenizer identifier.
        """
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
        """Yield the complete exported landscape graph once.

        Parameters
        ----------
        landscape : Any
            Landscape exposing ``to_graph_tensor`` or a graph and sequences.
        batch_size : int
            Accepted for interface compatibility and otherwise ignored.
        num_workers : int, default=0
            Accepted for interface compatibility and otherwise ignored.
        device : torch.device or None, optional
            Accepted for interface compatibility; transfer occurs later.
        **kwargs : Any
            Ignored compatibility options.

        Yields
        ------
        Any
            Full PyTorch Geometric-style graph object.

        Raises
        ------
        ImportError
            If fallback graph conversion requires PyTorch Geometric.
        RuntimeError
            If no graph export or fallback graph is available.
        ValueError
            If native graph export fails for a reason other than the supported
            variable-length fallback.
        """
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
        """Move a graph batch to the requested device when supported.

        Parameters
        ----------
        batch : Any
            Graph-like object.
        device : torch.device or None, optional
            Target device.

        Returns
        -------
        Any
            Device-mapped graph, or the original object when transfer is not
            supported or requested.
        """
        if device is not None and hasattr(batch, "to"):
            return batch.to(device)
        return batch


class NodeIndexInputAdapter(LandscapeInputAdapter):
    """Expose canonical landscape node positions as model inputs.

    Notes
    -----
    Indices are emitted as ``float32`` column tensors with shape
    ``(batch, 1)`` for compatibility with GP model inputs.
    """

    name = "node_index"

    def metadata(self, landscape: Any) -> Mapping[str, Any]:  # noqa: ARG002
        """Describe node-index input provenance.

        Parameters
        ----------
        landscape : Any
            Source landscape; accepted for interface consistency.

        Returns
        -------
        mapping of str to Any
            Adapter name and graph-node-index marker.
        """
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
        """Yield canonical node-index batches.

        Parameters
        ----------
        landscape : Any
            Landscape exposing ``sequences`` or a length.
        batch_size : int
            Number of node positions per batch.
        num_workers : int, default=0
            DataLoader worker-process count.
        device : torch.device or None, optional
            Accepted for interface compatibility; transfer occurs later.
        **kwargs : Any
            Ignored compatibility options.

        Yields
        ------
        torch.Tensor
            CPU ``float32`` column tensor of node indices.

        Raises
        ------
        RuntimeError
            If the landscape exposes no sequence count or length.
        """
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
        """Extract a node-index tensor and optionally move it to a device.

        Parameters
        ----------
        batch : Any
            Tensor or single-item DataLoader batch.
        device : torch.device or None, optional
            Target device.

        Returns
        -------
        Any
            Device-mapped tensor, or the original value when it is not a
            tensor.
        """
        tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        if device is not None and torch.is_tensor(tensor):
            return tensor.to(device)
        return tensor


InputAdapterFactory = Callable[..., LandscapeInputAdapter]
_INPUT_ADAPTERS: dict[str, InputAdapterFactory] = {}


def register_input_adapter(
    name: str, factory: InputAdapterFactory, *, overwrite: bool = False
) -> None:
    """Register a landscape input-adapter factory by name.

    Parameters
    ----------
    name : str
        Registry key.
    factory : callable
        Factory accepting adapter-specific keyword arguments.
    overwrite : bool, default=False
        Replace an existing registration.

    Returns
    -------
    None
        The process-local registry is mutated.

    Raises
    ------
    ValueError
        If ``name`` is registered and ``overwrite`` is false.
    """
    if name in _INPUT_ADAPTERS and not overwrite:
        raise ValueError(f"Input adapter '{name}' is already registered.")
    _INPUT_ADAPTERS[name] = factory


def resolve_input_adapter(
    adapter: LandscapeInputAdapter | str | None, **kwargs: Any
) -> LandscapeInputAdapter:
    """Resolve an input-adapter instance or registered name.

    Parameters
    ----------
    adapter : LandscapeInputAdapter, str, or None
        Existing adapter, registered name, or ``None`` for ``embedding``.
    **kwargs : Any
        Keyword arguments passed to the registered factory.

    Returns
    -------
    LandscapeInputAdapter
        Existing or newly constructed adapter.

    Raises
    ------
    ValueError
        If a requested registry name is unknown.
    """
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
    """Define conversion from named tensors to a Landscapy fitness layer.

    Attributes
    ----------
    layer_kind : str
        Logical output kind served by the adapter.
    """

    layer_kind: str

    @abstractmethod
    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        """Construct a fitness layer from model outputs.

        Parameters
        ----------
        outputs : mapping of str to torch.Tensor
            Named prediction tensors.
        categories : sequence of str or None
            Optional categorical label order.
        metadata : mapping of str to Any
            Prediction provenance attached to the new layer.
        layer_name : str
            Requested output layer name.

        Returns
        -------
        Any
            Landscapy fitness layer or compatible custom output.
        """
        ...


class FunctionOutputAdapter(LandscapeOutputAdapter):
    """Wrap a function as a landscape output adapter.

    Parameters
    ----------
    layer_kind : str
        Logical registry key assigned to the wrapper.
    func : callable
        Function accepting ``outputs``, ``categories``, ``metadata``, and
        ``layer_name`` in that order.

    Attributes
    ----------
    layer_kind : str
        Logical output kind.
    """

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
        """Delegate fitness-layer construction to the wrapped function.

        Parameters
        ----------
        outputs : mapping of str to torch.Tensor
            Named prediction tensors.
        categories : sequence of str or None
            Optional category order.
        metadata : mapping of str to Any
            Prediction provenance.
        layer_name : str
            Requested output layer name.

        Returns
        -------
        Any
            Value returned by the wrapped function.
        """
        return self._func(outputs, categories, metadata, layer_name)


class ProbCategoricalOutputAdapter(LandscapeOutputAdapter):
    """Build probabilistic categorical Landscapy fitness layers.

    Attributes
    ----------
    layer_kind : str
        Registry key ``"prob_categorical"``.
    """

    layer_kind = "prob_categorical"

    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        """Convert class probabilities and variance to a fitness layer.

        Parameters
        ----------
        outputs : mapping of str to torch.Tensor
            Required ``mean`` probability tensor with shape
            ``(n_sequences, n_categories)`` and optional same-shaped ``var``.
            Tensors may require gradients and reside on any PyTorch device.
        categories : sequence of str or None
            Ordered category names. Defaults to ``class_0``, ``class_1``, and
            so on.
        metadata : mapping of str to Any
            Prediction provenance. Variance is added as a NumPy array when
            supplied.
        layer_name : str
            Name for the created fitness layer.

        Returns
        -------
        ProbabilisticCategoricalFitness
            Layer containing detached CPU NumPy probabilities with the input
            tensor's dtype and shape.

        Raises
        ------
        ImportError
            If Landscapy is unavailable.
        TypeError
            If ``mean`` or ``var`` is not a tensor.
        ValueError
            If ``mean`` is absent or not two-dimensional, variance shape is
            incompatible, or category count differs from the output width.
        """
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
        if not torch.is_tensor(mean):
            raise TypeError(
                "Probabilistic categorical adapter received non-tensor 'mean' output."
            )
        if mean.ndim != 2:
            raise ValueError(
                "Probabilistic categorical adapter expects 'mean' with shape "
                "(n_sequences, n_categories)."
            )
        if var is not None:
            if not torch.is_tensor(var):
                raise TypeError(
                    "Probabilistic categorical adapter received non-tensor 'var' output."
                )
            if var.shape != mean.shape:
                raise ValueError(
                    "Probabilistic categorical adapter expects 'var' to have the "
                    "same shape as 'mean'."
                )
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
            meta["variance"] = var.detach().cpu().numpy()
        return ProbabilisticCategoricalFitness(
            name=layer_name,
            probabilities=mean.detach().cpu().numpy(),
            categories=cats,
            metadata=meta,
        )


class NumericOutputAdapter(LandscapeOutputAdapter):
    """Build numeric Landscapy fitness layers from tensor predictions.

    Attributes
    ----------
    layer_kind : str
        Registry key ``"numeric"``.
    """

    layer_kind = "numeric"

    def to_layer(
        self,
        outputs: Mapping[str, torch.Tensor],
        categories: Optional[Sequence[str]],  # noqa: ARG002 - not used for numeric layers
        metadata: Mapping[str, Any],
        layer_name: str,
    ) -> Any:
        """Convert numeric predictions to a detached CPU fitness layer.

        Parameters
        ----------
        outputs : mapping of str to torch.Tensor
            ``output`` tensor or a mapping containing exactly one tensor.
            Scalars, vectors, and matrices are accepted; scalars and vectors
            become two-dimensional column tensors.
        categories : sequence of str or None
            Ignored; accepted for the common output-adapter interface.
        metadata : mapping of str to Any
            Prediction provenance copied onto the layer.
        layer_name : str
            Name for the created fitness layer.

        Returns
        -------
        NumericFitness
            Numeric layer backed by a detached CPU tensor or values list,
            depending on the installed Landscapy API.

        Raises
        ------
        ImportError
            If Landscapy is unavailable.
        TypeError
            If the selected output is not a tensor.
        ValueError
            If no output is available or the tensor has more than two
            dimensions.
        """
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
    """Register an output-adapter instance or class by layer kind.

    Parameters
    ----------
    kind : str
        Logical output-layer key.
    adapter : LandscapeOutputAdapter or type of LandscapeOutputAdapter
        Reusable instance or zero-argument adapter class.
    overwrite : bool, default=False
        Replace an existing registration.

    Returns
    -------
    None
        The process-local registry is mutated.

    Raises
    ------
    TypeError
        If ``adapter`` is neither an adapter instance nor subclass.
    ValueError
        If ``kind`` is registered and ``overwrite`` is false.
    """
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
    """Register a function-style output adapter.

    Parameters
    ----------
    kind : str
        Logical output-layer key.
    adapter : callable
        Function accepting outputs, categories, metadata, and layer name.
    overwrite : bool, default=False
        Replace an existing registration.

    Returns
    -------
    None
        The function is wrapped and registered process-locally.

    Raises
    ------
    ValueError
        If ``kind`` is registered and ``overwrite`` is false.
    """
    output_adapter = FunctionOutputAdapter(kind, adapter)
    register_output_adapter(kind, output_adapter, overwrite=overwrite)


def resolve_output_adapter(kind: str) -> LandscapeOutputAdapter:
    """Construct the output adapter registered for a logical layer kind.

    Parameters
    ----------
    kind : str
        Logical output-layer key.

    Returns
    -------
    LandscapeOutputAdapter
        Newly constructed or registered reusable adapter.

    Raises
    ------
    ValueError
        If no adapter is registered for ``kind``.
    """
    if kind not in _OUTPUT_ADAPTERS:
        raise ValueError(f"No adapter registered for layer kind '{kind}'.")
    return _OUTPUT_ADAPTERS[kind]()


register_output_adapter(
    NumericOutputAdapter.layer_kind, NumericOutputAdapter, overwrite=True
)
register_output_adapter(
    ProbCategoricalOutputAdapter.layer_kind, ProbCategoricalOutputAdapter, overwrite=True
)
