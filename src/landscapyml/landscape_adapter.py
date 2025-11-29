from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from .data import SequenceClassificationDataModule


def _extract_label_mapping(layer: Any) -> Optional[list[str]]:
    cats = getattr(layer, "categories", None)
    if cats is None:
        meta = getattr(layer, "metadata", None)
        if isinstance(meta, Mapping):
            cats = meta.get("categories")
    if cats is None:
        return None
    return list(cats)


def records_from_landscape(
    landscape: Any,
    *,
    label_layer: str,
    label_key: Optional[str] = None,
    feature_view: str = "embedding",
    include_embeddings: bool = True,
    tokenizer: Any | str | None = None,
    sequence_idx: Optional[Sequence[int]] = None,
    sequence: Optional[Sequence[str] | str] = None,
) -> Tuple[list[dict[str, Any]], Optional[list[str]]]:
    """
    Convert a ``FitnessLandscape`` into classification-ready records.
    """
    if not hasattr(landscape, "to_sequence_tensors"):
        raise ValueError("Landscape must implement to_sequence_tensors.")

    key = label_key or label_layer
    raw = landscape.to_sequence_tensors(
        sequence_idx=sequence_idx,
        sequence=sequence,
        tokenizer=tokenizer,
        feature_view=feature_view,
        include_embeddings=include_embeddings,
        as_batch=False,
    )
    if not isinstance(raw, Sequence):
        raise ValueError("to_sequence_tensors must return a sequence of record mappings.")

    records: list[dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, Mapping):
            raise ValueError("Landscape export must yield mapping records.")
        fitness = rec.get("fitness_tensors") or {}
        if label_layer not in fitness:
            raise ValueError(f"Fitness layer '{label_layer}' missing from landscape export.")
        new_rec: dict[str, Any] = {
            "sequence_tensor": rec.get("sequence_tensor"),
            "fitness_tensors": {key: fitness[label_layer]},
        }
        if "embedding" in rec:
            new_rec["embedding"] = rec["embedding"]
        if "attention_mask" in rec:
            new_rec["attention_mask"] = rec["attention_mask"]
        records.append(new_rec)

    label_mapping = None
    layers = getattr(landscape, "fitness_layers", None)
    if isinstance(layers, Mapping):
        label_mapping = _extract_label_mapping(layers.get(label_layer))

    return records, label_mapping


def datamodule_from_landscape(
    landscape: Any,
    *,
    label_layer: str,
    label_key: Optional[str] = None,
    feature_view: str = "embedding",
    include_embeddings: bool = True,
    tokenizer: Any | str | None = None,
    **datamodule_kwargs: Any,
) -> SequenceClassificationDataModule:
    """
    Build a ``SequenceClassificationDataModule`` directly from a ``FitnessLandscape``.
    """
    records, label_mapping = records_from_landscape(
        landscape,
        label_layer=label_layer,
        label_key=label_key,
        feature_view=feature_view,
        include_embeddings=include_embeddings,
        tokenizer=tokenizer,
    )
    key = label_key or label_layer
    return SequenceClassificationDataModule(
        train_data=records,
        label_key=key,
        label_mapping=label_mapping,
        **datamodule_kwargs,
    )
