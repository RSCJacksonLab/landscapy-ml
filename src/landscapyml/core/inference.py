from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch

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
except Exception:  # pragma: no cover - optional dependency
    FitnessLandscape = Any  # type: ignore
    BaseFitnessLayer = Any  # type: ignore


def predict_sequences(
    model: Any,
    sequences: Sequence[Any],
    *,
    embedding_mode: str = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    embedding_batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed raw sequences and predict class probabilities with uncertainty.

    Parameters
    ----------
    model : Any
        Trained classifier implementing ``predict_with_uncertainty``.
    sequences : Sequence[Any]
        Raw sequences to embed and classify.
    embedding_mode : str, default="hard"
        Embedding strategy for raw sequences.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        Model identifier used by landscapy embedders.
    device : str, optional
        Device string forwarded to the embedder.
    embedding_batch_size : int, default=32
        Batch size used during embedding.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean class probabilities and variance tensors on CPU.
    """
    model.eval()
    embeddings, _, _ = embed_sequences(
        sequences,
        embedding_mode=embedding_mode,
        model_name=model_name,
        device=device,
        embedding_batch_size=embedding_batch_size,
        include_tokens=False,
    )
    embeddings = embeddings.to(model.device)
    mean_probs, variance = model.predict_with_uncertainty(embeddings)
    return mean_probs.cpu(), variance.cpu()


def predict_landscape_records(
    model: Any,
    records: Iterable[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on FitnessLandscape record dictionaries.

    Parameters
    ----------
    model : Any
        Trained classifier implementing ``predict_with_uncertainty``.
    records : Iterable[Mapping[str, Any]]
        Iterable of record dictionaries containing ``embedding`` or ``sequence_tensor``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean class probabilities and variance tensors on CPU.

    Raises
    ------
    ValueError
        If a record lacks the required feature fields.
    """
    model.eval()
    feats = []
    for rec in records:
        feature = rec.get("embedding", rec.get("sequence_tensor"))
        if feature is None:
            raise ValueError("Record missing 'embedding' or 'sequence_tensor'.")
        tensor = torch.as_tensor(feature, dtype=torch.float32)
        feats.append(tensor)
    inputs = torch.stack(feats, dim=0).to(model.device)
    mean_probs, variance = model.predict_with_uncertainty(inputs)
    return mean_probs.cpu(), variance.cpu()


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
) -> BaseFitnessLayer:
    """
    Run inference on a ``FitnessLandscape`` and construct a predicted fitness layer.

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
    device : str, optional
        Device to place model inputs on; defaults to the model device.
    attach : bool, default=True
        Whether to attach the resulting layer to the landscape.
    inplace : bool, default=True
        If ``attach`` is true, whether to attach in-place or on a copy.
    layer_name : str, default="predicted_fitness"
        Name assigned to the created layer.
    categories : Sequence[str], optional
        Optional category names; defaults to inferred class indices.
    input_adapter : LandscapeInputAdapter | str, optional
        Adapter used to extract model inputs from the landscape.
    input_adapter_kwargs : Mapping[str, Any], optional
        Optional kwargs used to construct the input adapter when a name is provided.

    Returns
    -------
    BaseFitnessLayer
        Fitness layer produced by the registered output adapter for the model's
        logical ``layer_kind``.

    Raises
    ------
    ValueError
        If the model type is unsupported or embedding compatibility checks fail.
    RuntimeError
        If the input adapter cannot supply model inputs.
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
        target = landscape if inplace else landscape.copy()
        target_layer_name = layer_name
        if hasattr(landscape, "safe_layer_name"):
            try:
                target_layer_name = landscape.safe_layer_name(layer_name)
            except Exception:
                target_layer_name = layer_name
        if hasattr(layer, "name"):
            layer.name = target_layer_name
        target.attach(layer=layer)
        return layer

    return layer
