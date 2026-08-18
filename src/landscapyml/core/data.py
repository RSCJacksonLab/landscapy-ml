from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .data_utils import (
    InputGetter,
    LandscapeRecord,
    TargetGetter,
    aggregate_numeric_targets,
    build_mask,
    expand_record_batch,
    feature_normalization_stats,
    make_fitness_target_getter,
    make_preferred_input_getter,
    normalize_records,
    resolve_split_indices,
    sequence_composition_features,
)

DatasetFactory = Callable[..., Dataset]
CollateFn = Callable[[list[Any]], Any]


class LandscapeDataset(Dataset):
    """
    Generic dataset for landscapy record dictionaries.

    Records preserve the landscapy boundary: a sequence view under
    ``sequence_tensor`` or ``embedding`` plus one or more labels in
    ``fitness_tensors``. Task-specific datasets provide only input and target
    getters; they do not need to know how the landscape was constructed.
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
    Generic Lightning ``DataModule`` for landscapy-derived record datasets.
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
        collate_fn: Optional[CollateFn] = None,
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
        try:
            self.val_split = float(val_split)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"val_split must satisfy 0 <= val_split < 1 "
                f"(got {val_split!r} for {len(self.train_records)} training records)."
            ) from exc
        if not np.isfinite(self.val_split) or not 0 <= self.val_split < 1:
            raise ValueError(
                f"val_split must satisfy 0 <= val_split < 1 "
                f"(got {val_split!r} for {len(self.train_records)} training records)."
            )
        self.val_seed = val_seed
        self.dataset_factory = dataset_factory
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.predict_dataset_kwargs = (
            dict(predict_dataset_kwargs)
            if predict_dataset_kwargs is not None
            else dict(self.dataset_kwargs)
        )
        self.collate_fn = collate_fn

        if not self.train_records:
            raise ValueError("train_data must not be empty.")
        if not self.val_records and self.val_split > 0:
            train_count = int(len(self.train_records) * (1 - self.val_split))
            if train_count < 1:
                raise ValueError(
                    f"val_split={self.val_split} leaves no training records "
                    f"from {len(self.train_records)} input records."
                )

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
                if not train_idx or not val_idx:
                    raise ValueError(
                        f"val_split={self.val_split} cannot produce non-empty "
                        f"training and validation partitions from {len(idx)} records."
                    )
                source_records = self.train_records
                new_train_records = [source_records[i] for i in train_idx]
                new_val_records = [source_records[i] for i in val_idx]
                self.train_records, self.val_records = (
                    new_train_records,
                    new_val_records,
                )

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
            collate_fn=self.collate_fn,
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


def _collate_single_graph(items: list[Any]) -> Any:
    return items[0]


def _graph_record_getter(record: Mapping[str, Any]) -> Any:
    if "graph" not in record:
        raise ValueError("Graph dataset records must contain a 'graph' item.")
    return record["graph"]


class LandscapeGraphDataset(LandscapeDataset):
    """
    Dataset specialization for single-graph landscape regression.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            records,
            input_getter=_graph_record_getter,
        )


def _graph_record(graph: Any | None) -> list[dict[str, Any]] | None:
    if graph is None:
        return None
    return [{"graph": graph}]


def _mask_has_observations(graph: Any, mask_name: str) -> bool:
    mask = getattr(graph, mask_name, None)
    return mask is not None and int(mask.sum().item()) > 0


def build_regression_graph_from_landscape(
    landscape: Any,
    *,
    target_layer: str,
    tokenizer: Any | str | None = None,
    aggregate_func: Optional[Callable[..., Any]] = np.mean,
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    val_indices: Sequence[int] | torch.Tensor | None = None,
    test_indices: Sequence[int] | torch.Tensor | None = None,
    seed: Optional[int] = None,
    normalize_features: bool = False,
    feature_normalization_eps: float = 1e-8,
) -> Any:
    """
    Build a PyG-style single-graph regression object from a ``FitnessLandscape``.

    Known numeric fitness values become supervised targets. Unknown values
    should be encoded as ``NaN`` in the source layer. Predefined
    train/validation/test indices can be supplied; otherwise finite targets are
    randomly split using ``val_fraction`` and ``test_fraction``.
    """

    if not hasattr(landscape, "to_graph_tensor") and not hasattr(landscape, "graph"):
        raise ValueError("Landscape must implement to_graph_tensor() or expose a graph.")
    layers = getattr(landscape, "fitness_layers", None)
    if not isinstance(layers, Mapping) or target_layer not in layers:
        raise ValueError(f"Landscape does not contain target layer '{target_layer}'.")

    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    if hasattr(landscape, "to_graph_tensor"):
        try:
            graph = landscape.to_graph_tensor(tokenizer=tokenizer)
        except ValueError as exc:
            if "inhomogeneous shape" not in str(exc):
                raise
            graph = _graph_tensor_from_landscape_graph(landscape)
    else:
        graph = _graph_tensor_from_landscape_graph(landscape)
    values = aggregate_numeric_targets(
        layers[target_layer],
        aggregate_func=aggregate_func,
    )
    y = torch.as_tensor(values, dtype=torch.float32).view(-1)
    if y.shape[0] != int(getattr(graph, "num_nodes", y.shape[0])):
        raise ValueError("Target layer length does not match graph node count.")

    known_idx = torch.nonzero(torch.isfinite(y), as_tuple=False).view(-1)
    if known_idx.numel() == 0:
        raise ValueError("No known numeric fitness values were found for training.")

    num_nodes = y.shape[0]
    train_idx, val_idx, test_idx = resolve_split_indices(
        num_nodes=num_nodes,
        known_idx=known_idx,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    graph.y = y
    graph.target_layer = target_layer
    graph.known_mask = build_mask(num_nodes, known_idx)
    graph.train_mask = build_mask(num_nodes, train_idx)
    graph.val_mask = build_mask(num_nodes, val_idx)
    graph.test_mask = build_mask(num_nodes, test_idx)
    graph.predict_mask = ~graph.known_mask
    graph.normalize_features = bool(normalize_features)
    if normalize_features:
        mean, scale = feature_normalization_stats(
            graph.x,
            mask=graph.train_mask,
            eps=feature_normalization_eps,
        )
        graph.feature_normalization_mean = mean
        graph.feature_normalization_scale = scale
    return graph


def _graph_tensor_from_landscape_graph(landscape: Any) -> Any:
    try:
        from torch_geometric.data import Data
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Graph tensor fallback requires torch-geometric to be installed."
        ) from exc

    source_graph = getattr(landscape, "graph", None)
    sequences = list(getattr(landscape, "sequences", []) or [])
    if source_graph is None or not sequences:
        raise ValueError("Landscape graph tensor fallback requires graph and sequences.")

    node_order = list(getattr(landscape, "_node_order", list(source_graph.nodes())))
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    embedding = landscape.get_embedding() if hasattr(landscape, "get_embedding") else None
    if embedding is not None and len(embedding) == len(node_order):
        x = torch.as_tensor(embedding, dtype=torch.float32)
    else:
        x = torch.as_tensor(
            sequence_composition_features(sequences),
            dtype=torch.float32,
        )

    edges: list[tuple[int, int]] = []
    for src, dst in source_graph.edges():
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        edges.append((src_idx, dst_idx))
        if not getattr(source_graph, "is_directed", lambda: False)():
            edges.append((dst_idx, src_idx))

    if edges:
        edge_index = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=len(node_order))


class LandscapeGraphRegressionDataModule(LandscapeDataModule):
    """
    DataModule for node-level regression on a single fitness landscape graph.
    """

    def __init__(
        self,
        *,
        train_graph: Any,
        val_graph: Any = None,
        test_graph: Any = None,
        predict_graph: Any = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.test_graph = test_graph
        self.predict_graph = predict_graph
        super().__init__(
            train_data=_graph_record(train_graph),
            val_data=_graph_record(val_graph),
            test_data=_graph_record(test_graph),
            predict_data=_graph_record(predict_graph),
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            dataset_factory=LandscapeGraphDataset,
            collate_fn=_collate_single_graph,
        )

    @classmethod
    def from_landscape(
        cls,
        *,
        landscape: Any,
        target_layer: str,
        tokenizer: Any | str | None = None,
        aggregate_func: Optional[Callable[..., Any]] = np.mean,
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        train_indices: Sequence[int] | torch.Tensor | None = None,
        val_indices: Sequence[int] | torch.Tensor | None = None,
        test_indices: Sequence[int] | torch.Tensor | None = None,
        seed: Optional[int] = None,
        normalize_features: bool = False,
        feature_normalization_eps: float = 1e-8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> "LandscapeGraphRegressionDataModule":
        graph = build_regression_graph_from_landscape(
            landscape,
            target_layer=target_layer,
            tokenizer=tokenizer,
            aggregate_func=aggregate_func,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            seed=seed,
            normalize_features=normalize_features,
            feature_normalization_eps=feature_normalization_eps,
        )
        return cls(
            train_graph=graph,
            val_graph=graph if _mask_has_observations(graph, "val_mask") else None,
            test_graph=graph if _mask_has_observations(graph, "test_mask") else None,
            predict_graph=graph,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


__all__ = [
    "CollateFn",
    "DatasetFactory",
    "InputGetter",
    "LandscapeDataModule",
    "LandscapeDataset",
    "LandscapeGraphDataset",
    "LandscapeGraphRegressionDataModule",
    "LandscapeRecord",
    "TargetGetter",
    "build_regression_graph_from_landscape",
    "expand_record_batch",
    "make_fitness_target_getter",
    "make_preferred_input_getter",
    "normalize_records",
]
