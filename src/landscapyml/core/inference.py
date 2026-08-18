"""Run model inference and convert predictions into Landscapy layers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from ._optional import is_missing_optional_dependency
from .adaptor import (
    LandscapeInputAdapter,
    ModelAdapter,
    infer_device,
    normalize_adapter_outputs,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    resolve_input_adapter,
    resolve_model_adapter,
    resolve_output_adapter,
)
from .data_utils import embed_sequences

try:
    from fitness_landscape.core.fitness import BaseFitnessLayer
    from fitness_landscape.core.landscape import FitnessLandscape
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if not is_missing_optional_dependency(exc, "fitness_landscape"):
        raise
    FitnessLandscape = Any  # type: ignore
    BaseFitnessLayer = Any  # type: ignore


@dataclass(frozen=True)
class LandscapeInferenceResult:
    """Hold a copied landscape and the prediction layer attached to it.

    Parameters
    ----------
    landscape : FitnessLandscape
        Independent landscape copy containing ``layer``.
    layer : BaseFitnessLayer
        Prediction layer attached to ``landscape``.

    Attributes
    ----------
    landscape : FitnessLandscape
        Independent landscape copy containing ``layer``.
    layer : BaseFitnessLayer
        Prediction layer attached to ``landscape``.
    """

    landscape: FitnessLandscape
    layer: BaseFitnessLayer


def _validated_uncertainty_outputs(
    outputs: Any, *, expected_batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise ValueError(
            "predict_with_uncertainty must return exactly two outputs: "
            "mean and variance."
        )

    mean_probs, variance = outputs
    if not torch.is_tensor(mean_probs) or not torch.is_tensor(variance):
        raise TypeError(
            "predict_with_uncertainty mean and variance outputs must be "
            "torch.Tensor objects."
        )
    if mean_probs.ndim == 0 or mean_probs.shape[0] != expected_batch_size:
        raise ValueError(
            "Mean prediction batch dimension does not match the number of inputs."
        )
    if variance.ndim == 0 or variance.shape[0] != expected_batch_size:
        raise ValueError(
            "Variance prediction batch dimension does not match the number of inputs."
        )
    if variance.shape != mean_probs.shape:
        raise ValueError(
            "Mean and variance prediction tensors must have the same shape."
        )

    return mean_probs.detach().cpu(), variance.detach().cpu()


def predict_sequences(
    model: Any,
    sequences: Sequence[Any],
    *,
    embedding_mode: str = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    embedding_batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed raw sequences and predict probabilities with uncertainty.

    Parameters
    ----------
    model : Any
        Trained classifier implementing ``predict_with_uncertainty``.
    sequences : sequence of Any
        Raw sequences to embed and classify.
    embedding_mode : str, default="hard"
        Embedding strategy for raw sequences.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        Model identifier used by landscapy embedders.
    device : str or None, optional
        Device forwarded to the embedder. Model execution uses the model's
        declared or first-parameter device, with a CPU fallback.
    embedding_batch_size : int, default=32
        Batch size used during embedding.

    Returns
    -------
    mean_probabilities : torch.Tensor
        Detached CPU prediction tensor retaining model output shape and dtype.
    variance : torch.Tensor
        Detached CPU uncertainty tensor with the same shape as the mean.

    Raises
    ------
    ValueError
        If ``sequences`` is empty or outputs have invalid count, shape, or batch
        dimension.
    TypeError
        If either model output is not a tensor.

    Notes
    -----
    The model is placed in evaluation mode and invoked under
    :func:`torch.inference_mode`. Raw-sequence embedding may download external
    model weights through Landscapy.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one input.")

    model.eval()
    model_device = infer_device(model) or torch.device("cpu")
    with torch.inference_mode():
        embeddings, _, _ = embed_sequences(
            sequences,
            embedding_mode=embedding_mode,
            model_name=model_name,
            device=device,
            embedding_batch_size=embedding_batch_size,
            include_tokens=False,
        )
        embeddings = embeddings.to(model_device)
        outputs = model.predict_with_uncertainty(embeddings)
    return _validated_uncertainty_outputs(
        outputs,
        expected_batch_size=embeddings.shape[0],
    )


def predict_landscape_records(
    model: Any,
    records: Iterable[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict probabilities from Landscapy record dictionaries.

    Parameters
    ----------
    model : Any
        Trained classifier implementing ``predict_with_uncertainty``.
    records : iterable of mappings
        Iterable of record dictionaries containing ``embedding`` or ``sequence_tensor``.

    Returns
    -------
    mean_probabilities : torch.Tensor
        Detached CPU prediction tensor retaining model output shape and dtype.
    variance : torch.Tensor
        Detached CPU uncertainty tensor with the same shape as the mean.

    Raises
    ------
    ValueError
        If no records are supplied, a record lacks the required feature fields,
        feature shapes differ, or the model returns incompatible outputs.
    TypeError
        If a record is not a mapping or either model output is not a tensor.

    Notes
    -----
    Record features are converted to ``float32`` and stacked on a new leading
    batch axis. The model's declared or first-parameter device is used, with a
    CPU fallback, and prediction runs under :func:`torch.inference_mode`.
    """
    model.eval()
    feats: list[torch.Tensor] = []
    expected_shape: Optional[torch.Size] = None
    for index, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"Record {index} must be a mapping.")
        feature = rec.get("embedding")
        if feature is None:
            feature = rec.get("sequence_tensor")
        if feature is None:
            raise ValueError(
                f"Record {index} is missing 'embedding' or 'sequence_tensor'."
            )
        tensor = torch.as_tensor(feature, dtype=torch.float32)
        if expected_shape is None:
            expected_shape = tensor.shape
        elif tensor.shape != expected_shape:
            raise ValueError(
                f"Record {index} feature shape {tuple(tensor.shape)} does not match "
                f"the expected shape {tuple(expected_shape)}."
            )
        feats.append(tensor)
    if not feats:
        raise ValueError("records must contain at least one input.")

    model_device = infer_device(model) or torch.device("cpu")
    inputs = torch.stack(feats, dim=0).to(model_device)
    with torch.inference_mode():
        outputs = model.predict_with_uncertainty(inputs)
    return _validated_uncertainty_outputs(
        outputs,
        expected_batch_size=inputs.shape[0],
    )


def infer_fitness_layer_from_landscape(
    landscape: FitnessLandscape,
    model: Any,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[str] = None,
    attach: bool = True,
    inplace: bool = True,
    layer_name: str = "predicted_fitness",
    categories: Optional[Sequence[str]] = None,
    input_adapter: LandscapeInputAdapter | str | None = None,
    input_adapter_kwargs: Optional[Mapping[str, Any]] = None,
) -> BaseFitnessLayer | LandscapeInferenceResult:
    """Infer and optionally attach a predicted fitness layer.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape object providing embeddings.
    model : Any
        Trained model or ``ModelAdapter`` compatible with the registry.
    batch_size : int, default=256
        Batch size for inference DataLoader.
    num_workers : int, default=0
        Number of DataLoader workers.
    device : str or None, optional
        Device to place model inputs on; defaults to the model device.
    attach : bool, default=True
        Whether to attach the resulting layer to the landscape.
    inplace : bool, default=True
        If ``attach`` is true, attach to the supplied landscape when true or an
        independent copy when false. This option has no effect when ``attach``
        is false.
    layer_name : str, default="predicted_fitness"
        Name assigned to the created layer.
    categories : sequence of str or None, optional
        Optional category names; defaults to inferred class indices.
    input_adapter : LandscapeInputAdapter, str, or None, optional
        Adapter used to extract model inputs from the landscape.
    input_adapter_kwargs : mapping of str to Any or None, optional
        Optional kwargs used to construct the input adapter when a name is provided.

    Returns
    -------
    BaseFitnessLayer or LandscapeInferenceResult
        The fitness layer when ``attach`` is false or ``inplace`` is true. For
        ``attach=True, inplace=False``, return the copied landscape together
        with its attached layer in a ``LandscapeInferenceResult``.

    Raises
    ------
    ValueError
        If the model type is unsupported or embedding compatibility checks fail.
    RuntimeError
        If the input adapter cannot supply model inputs.

    Notes
    -----
    Prediction executes without gradient recording. Output batches are
    detached and moved to CPU before layer conversion. With in-place
    attachment, ``landscape`` is mutated and the layer is returned. Copy-mode
    attachment deep-copies the landscape, leaves the original unchanged, and
    returns both objects in ``LandscapeInferenceResult``. Collision-safe layer
    naming is resolved on the actual attachment target.
    """
    model_adapter = resolve_model_adapter(model)
    layer_kind = model_adapter.layer_kind
    layer_adapter = resolve_output_adapter(layer_kind)
    input_adapter_obj = resolve_input_adapter(
        input_adapter, **(input_adapter_kwargs or {})
    )

    expected_domain = getattr(model_adapter, "embedding_domain", None)
    expected_model = getattr(model_adapter, "embedding_model", None)
    active_domain, active_model, _ = input_adapter_obj.embedding_info(landscape)
    if expected_domain and active_domain and expected_domain != active_domain:
        raise ValueError(
            f"Embedding domain mismatch: model expects {expected_domain}, landscape active is {active_domain}."
        )
    if expected_model and active_model and expected_model != active_model:
        raise ValueError(
            f"Embedding model mismatch: model expects {expected_model}, landscape has {active_model}."
        )

    outputs_by_key: dict[str, list[torch.Tensor]] = {}
    if hasattr(model_adapter, "eval"):
        model_adapter.eval()
    model_device = (
        torch.device(device)
        if device is not None
        else infer_device(model_adapter)
        or infer_device(getattr(model_adapter, "model", None))
        or infer_device(model)
        or torch.device("cpu")
    )
    if device is not None and hasattr(model_adapter, "to"):
        model_adapter.to(model_device)
    with torch.no_grad():
        for batch in input_adapter_obj.iter_batches(
            landscape,
            batch_size=batch_size,
            num_workers=num_workers,
            device=model_device,
            model_name=expected_model,
        ):
            inputs = input_adapter_obj.to_model_inputs(batch, device=model_device)
            batch_outputs = normalize_adapter_outputs(
                model_adapter.predict(inputs), layer_kind
            )
            for key, tensor in batch_outputs.items():
                if tensor is None:
                    continue
                if not torch.is_tensor(tensor):
                    raise ValueError(
                        f"Adapter output '{key}' must be a torch.Tensor."
                    )
                outputs_by_key.setdefault(key, []).append(tensor.detach().cpu())

    outputs = {
        key: torch.cat(tensors, dim=0) for key, tensors in outputs_by_key.items()
    }

    model_meta = getattr(model_adapter, "model", None) or model
    model_type = type(model_meta)

    final_domain, final_model, final_meta = input_adapter_obj.embedding_info(
        landscape
    )
    meta = {
        "predicted": True,
        "model_type": model_type.__name__,
        "layer_kind": layer_kind,
    }
    meta.update(input_adapter_obj.metadata(landscape))
    if model_adapter is not model:
        meta["adapter_type"] = type(model_adapter).__name__
    if "embedding_domain" not in meta and (final_domain or expected_domain):
        meta["embedding_domain"] = final_domain or expected_domain
    if "embedding_model" not in meta and (final_model or expected_model):
        meta["embedding_model"] = final_model or expected_model
    if (
        "embedding_mode" not in meta
        and isinstance(final_meta, Mapping)
        and "embedding_mode" in final_meta
    ):
        meta["embedding_mode"] = final_meta["embedding_mode"]
    layer = layer_adapter.to_layer(outputs, categories, meta, layer_name)

    if attach and hasattr(landscape, "attach"):
        target = landscape if inplace else deepcopy(landscape)
        target_layer_name = layer_name
        if hasattr(target, "safe_layer_name"):
            target_layer_name = target.safe_layer_name(layer_name)
        if hasattr(layer, "name"):
            layer.name = target_layer_name
        target.attach(layer=layer)
        if not inplace:
            return LandscapeInferenceResult(landscape=target, layer=layer)
        return layer

    return layer
