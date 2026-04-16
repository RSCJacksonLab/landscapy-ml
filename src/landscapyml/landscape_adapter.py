from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .data import SequenceClassificationDataModule
from .landscape_pipeline import export_landscape_records


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
    key = label_key or label_layer
    exported = export_landscape_records(
        landscape,
        fitness_layers=[label_layer],
        rename_fitness={label_layer: key},
        sequence_idx=sequence_idx,
        sequence=sequence,
        tokenizer=tokenizer,
        feature_view=feature_view,
        include_embeddings=include_embeddings,
    )
    return exported.records, exported.fitness_mappings.get(key)


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
