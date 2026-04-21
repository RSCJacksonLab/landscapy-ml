from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

LandscapeRecord = dict[str, Any]
InputGetter = Callable[[Mapping[str, Any]], Any]
TargetGetter = Callable[[Mapping[str, Any]], Any]
DatasetFactory = Callable[..., Dataset]


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
    Export a ``FitnessLandscape`` into ML-ready record dictionaries.

    Unlike the legacy helpers, this function does not assume a single
    classification label. It preserves all requested fitness layers and
    therefore acts as the package's task-agnostic interface boundary.
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


def expand_record_batch(batch: Mapping[str, Any]) -> list[LandscapeRecord]:
    """
    Convert a batched landscape export into per-record dictionaries.
    """

    if "sequence_tensor" not in batch or "fitness_tensors" not in batch:
        raise ValueError(
            "Batch dictionary must contain 'sequence_tensor' and 'fitness_tensors'."
        )

    seqs = torch.as_tensor(batch["sequence_tensor"])
    fitness = {k: torch.as_tensor(v) for k, v in batch["fitness_tensors"].items()}
    attention_mask = batch.get("attention_mask")
    attention_mask_t = (
        torch.as_tensor(attention_mask) if attention_mask is not None else None
    )
    embedding = batch.get("embedding")
    embedding_t = torch.as_tensor(embedding) if embedding is not None else None

    records: list[LandscapeRecord] = []
    for idx in range(seqs.shape[0]):
        record: LandscapeRecord = {
            "sequence_tensor": seqs[idx],
            "fitness_tensors": {name: tensor[idx] for name, tensor in fitness.items()},
        }
        if attention_mask_t is not None:
            record["attention_mask"] = attention_mask_t[idx]
        if embedding_t is not None:
            record["embedding"] = embedding_t[idx]
        records.append(record)
    return records


def normalize_records(data: Any) -> list[LandscapeRecord]:
    """
    Normalize supported landscape record inputs into a list of records.
    """

    if data is None:
        return []
    if isinstance(data, Mapping):
        return expand_record_batch(data)
    if isinstance(data, Iterable) and not isinstance(
        data, (str, bytes, bytearray, torch.Tensor)
    ):
        items = list(data)
        if not items:
            return []
        if isinstance(items[0], Mapping):
            return [dict(item) for item in items]
    raise ValueError("Data must be a batch dict or an iterable of record dictionaries.")


def make_preferred_input_getter(
    *feature_keys: str, cast_float: bool = True
) -> InputGetter:
    keys = feature_keys or ("embedding", "sequence_tensor")

    def _getter(record: Mapping[str, Any]) -> torch.Tensor:
        for key in keys:
            feature = record.get(key)
            if feature is None:
                continue
            tensor = torch.as_tensor(feature)
            if cast_float and not tensor.is_floating_point():
                tensor = tensor.float()
            return tensor
        key_list = ", ".join(keys)
        raise ValueError(f"Record missing feature view. Tried: {key_list}.")

    return _getter


def make_fitness_target_getter(
    layer_name: str,
    *,
    collapse_one_hot: bool = False,
    dtype: Optional[torch.dtype] = None,
    squeeze: bool = True,
) -> TargetGetter:
    def _getter(record: Mapping[str, Any]) -> torch.Tensor:
        fitness = record.get("fitness_tensors")
        if not isinstance(fitness, Mapping) or layer_name not in fitness:
            raise ValueError(f"Record missing fitness label '{layer_name}'.")

        target = torch.as_tensor(fitness[layer_name])
        if collapse_one_hot and target.ndim > 0 and target.numel() > 1:
            target = target.argmax(dim=-1)
        if dtype is not None:
            target = target.to(dtype=dtype)
        if squeeze:
            target = target.squeeze()
        return target

    return _getter


class LandscapeDataset(Dataset):
    """
    Generic dataset for wiring landscape records into ML pipelines.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        input_getter: Optional[InputGetter] = None,
        target_getter: Optional[TargetGetter] = None,
        return_format: str = "tuple",
    ) -> None:
        if not records:
            raise ValueError("records must be a non-empty sequence.")
        if return_format not in {"tuple", "dict"}:
            raise ValueError("return_format must be 'tuple' or 'dict'.")
        self.records = [dict(record) for record in records]
        self.input_getter = input_getter or make_preferred_input_getter()
        self.target_getter = target_getter
        self.return_format = return_format

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        inputs = self.input_getter(record)

        if self.target_getter is None:
            if self.return_format == "dict":
                return {"inputs": inputs, "record": record}
            return inputs

        targets = self.target_getter(record)
        if self.return_format == "dict":
            return {"inputs": inputs, "targets": targets, "record": record}
        return inputs, targets


class LandscapeDataModule(pl.LightningDataModule):
    """
    Generic Lightning ``DataModule`` for landscape-derived record datasets.
    """

    def __init__(
        self,
        *,
        train_data: Any,
        val_data: Any = None,
        test_data: Any = None,
        predict_data: Any = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        val_split: float = 0.0,
        val_seed: Optional[int] = None,
        dataset_factory: DatasetFactory = LandscapeDataset,
        dataset_kwargs: Optional[Mapping[str, Any]] = None,
        predict_dataset_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.train_records = normalize_records(train_data)
        self.val_records = normalize_records(val_data) if val_data is not None else []
        self.test_records = normalize_records(test_data) if test_data is not None else []
        self.predict_records = (
            normalize_records(predict_data) if predict_data is not None else []
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.val_split = val_split
        self.val_seed = val_seed
        self.dataset_factory = dataset_factory
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.predict_dataset_kwargs = (
            dict(predict_dataset_kwargs)
            if predict_dataset_kwargs is not None
            else dict(self.dataset_kwargs)
        )

        if not self.train_records:
            raise ValueError("train_data must not be empty.")

        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None
        self._predict_ds: Optional[Dataset] = None

    def _build_dataset(
        self, records: Sequence[Mapping[str, Any]], *, stage: str
    ) -> Optional[Dataset]:
        if not records:
            return None
        kwargs = (
            self.predict_dataset_kwargs if stage == "predict" else self.dataset_kwargs
        )
        return self.dataset_factory(records, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if not self.val_records and self.val_split > 0:
                rng = torch.Generator().manual_seed(self.val_seed or 0)
                idx = torch.randperm(len(self.train_records), generator=rng).tolist()
                split = int(len(idx) * (1 - self.val_split))
                train_idx, val_idx = idx[:split], idx[split:]
                self.val_records = [self.train_records[i] for i in val_idx]
                self.train_records = [self.train_records[i] for i in train_idx]

            self._train_ds = self._build_dataset(self.train_records, stage="fit")
            self._val_ds = self._build_dataset(self.val_records, stage="fit")

        if stage in (None, "test"):
            self._test_ds = self._build_dataset(self.test_records, stage="test")

        if stage in (None, "predict"):
            self._predict_ds = self._build_dataset(
                self.predict_records, stage="predict"
            )

    def _loader(
        self, dataset: Optional[Dataset], *, shuffle: bool = False
    ) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        loader = self._loader(self._train_ds, shuffle=self.shuffle)
        if loader is None:
            raise RuntimeError("Training dataset was not initialized.")
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._val_ds)

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._test_ds)

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self._predict_ds)


__all__ = [
    "LandscapeDataModule",
    "LandscapeDataset",
    "LandscapeExport",
    "export_landscape_records",
    "expand_record_batch",
    "make_fitness_target_getter",
    "make_preferred_input_getter",
    "normalize_records",
]
