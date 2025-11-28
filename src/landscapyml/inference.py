from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Type

import torch

from .data import embed_sequences
from .gp_classification import SequenceGPClassifier
from .mlp_classification import SequenceMLPEnsembleClassifier

try:
    from fitness_landscape.core.fitness import ProbabilisticCategoricalFitness
    from fitness_landscape.core.landscape import FitnessLandscape
except Exception:  # pragma: no cover - optional dependency
    FitnessLandscape = Any  # type: ignore
    ProbabilisticCategoricalFitness = Any  # type: ignore


def predict_sequences(
    model: SequenceGPClassifier,
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
    model : SequenceGPClassifier
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
    model: SequenceGPClassifier,
    records: Iterable[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on FitnessLandscape record dictionaries.

    Parameters
    ----------
    model : SequenceGPClassifier
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


_MODEL_TO_LAYER: dict[Type[Any], str] = {
    SequenceGPClassifier: "prob_categorical",
    SequenceMLPEnsembleClassifier: "prob_categorical",
}

LayerAdapter = Callable[
    [Mapping[str, torch.Tensor], Optional[Sequence[str]], Mapping[str, Any], str],
    Any,
]


def _prob_categorical_adapter(
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
        raise ValueError("Number of categories does not match model output dimension.")
    meta = dict(metadata)
    if var is not None:
        meta["variance"] = var.numpy()
    return ProbabilisticCategoricalFitness(
        name=layer_name,
        probabilities=mean.numpy(),
        categories=cats,
        metadata=meta,
    )


_LAYER_ADAPTERS: dict[str, LayerAdapter] = {
    "prob_categorical": _prob_categorical_adapter,
}


def register_model_layer_mapping(
    model_cls: Type[Any], layer_kind: str, *, overwrite: bool = False
) -> None:
    if model_cls in _MODEL_TO_LAYER and not overwrite:
        raise ValueError(
            f"Model {model_cls.__name__} already mapped to {_MODEL_TO_LAYER[model_cls]!r}."
        )
    _MODEL_TO_LAYER[model_cls] = layer_kind


def register_layer_adapter(
    kind: str, adapter: LayerAdapter, *, overwrite: bool = False
) -> None:
    if kind in _LAYER_ADAPTERS and not overwrite:
        raise ValueError(f"Adapter for layer kind {kind!r} already exists.")
    _LAYER_ADAPTERS[kind] = adapter


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
) -> ProbabilisticCategoricalFitness:
    """
    Run inference on a ``FitnessLandscape`` and construct a predicted fitness layer.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape object providing embeddings.
    model : Any
        Trained model registered in ``_MODEL_TO_LAYER``.
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

    Returns
    -------
    ProbabilisticCategoricalFitness
        Fitness layer produced by the registered adapter.

    Raises
    ------
    ValueError
        If the model type is unsupported or embedding compatibility checks fail.
    RuntimeError
        If embeddings are unavailable and cannot be computed.
    """
    model_type = type(model)
    layer_kind = _MODEL_TO_LAYER.get(model_type)
    if layer_kind is None:
        raise ValueError(
            f"Model type {model_type.__name__} is not supported for inference."
        )
    adapter = _LAYER_ADAPTERS.get(layer_kind)
    if adapter is None:
        raise ValueError(f"No adapter registered for layer kind '{layer_kind}'.")

    expected_domain = getattr(model, "embedding_domain", None)
    expected_model = getattr(model, "embedding_model", None)
    active_domain = getattr(landscape, "_active_embedding_domain", None)
    active_model = getattr(landscape, "embedding_model", None)
    if expected_domain and active_domain and expected_domain != active_domain:
        raise ValueError(
            f"Embedding domain mismatch: model expects {expected_domain}, landscape active is {active_domain}."
        )
    if expected_model and active_model and expected_model != active_model:
        raise ValueError(
            f"Embedding model mismatch: model expects {expected_model}, landscape has {active_model}."
        )

    emb_array = landscape.get_embedding()
    if emb_array is None:
        model_name = expected_model or "facebook/esm2_t6_8M_UR50D"
        landscape.compute_plm_embeddings(model_name=model_name)
        emb_array = landscape.get_embedding()
    if emb_array is None:
        raise RuntimeError(
            "No embeddings available on landscape and automatic computation failed."
        )

    emb = torch.as_tensor(emb_array, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(emb, torch.zeros(len(emb), dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    outputs: dict[str, torch.Tensor] = {}
    model.eval()
    model_device = device or next(model.parameters()).device
    if layer_kind == "prob_categorical":
        if not hasattr(model, "predict_with_uncertainty"):
            raise ValueError(
                f"Model {model_type.__name__} must implement predict_with_uncertainty."
            )
        all_mean: list[torch.Tensor] = []
        all_var: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(model_device)
                mean_probs, var_probs = model.predict_with_uncertainty(xb)
                all_mean.append(mean_probs.cpu())
                all_var.append(var_probs.cpu())
        outputs["mean"] = torch.cat(all_mean, dim=0)
        outputs["var"] = torch.cat(all_var, dim=0)
    else:
        # Generic forward pass
        all_out: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(model_device)
                out = model(xb)
                all_out.append(out.cpu())
        outputs["output"] = torch.cat(all_out, dim=0)

    meta = {
        "predicted": True,
        "model_type": model_type.__name__,
        "layer_kind": layer_kind,
        "embedding_domain": active_domain or expected_domain,
        "embedding_model": active_model or expected_model,
    }
    layer = adapter(outputs, categories, meta, layer_name)

    if attach and hasattr(landscape, "attach"):
        target = landscape if inplace else landscape.copy()
        target.attach(layer=layer)
        return layer

    return layer
