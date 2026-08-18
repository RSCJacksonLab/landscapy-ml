"""PyTorch datasets and Lightning data modules for Landscapy records."""

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
    """Represent Landscapy record dictionaries as a PyTorch dataset.

    Records preserve the landscapy boundary: a sequence view under
    ``sequence_tensor`` or ``embedding`` plus one or more labels in
    ``fitness_tensors``. Task-specific datasets provide only input and target
    getters; they do not need to know how the landscape was constructed.

    Parameters
    ----------
    records : sequence of mappings
        Non-empty per-sequence record collection.
    input_getter : callable or None, optional
        Function extracting model inputs. The default prefers embeddings over
        sequence tensors.
    target_getter : callable or None, optional
        Function extracting supervised targets. Omit for prediction-only data.
    return_format : {"tuple", "dict"}, default="tuple"
        Return inputs and targets as tuples or named dictionary entries.

    Attributes
    ----------
    records : list of dict
        Shallow copies of the supplied record mappings.
    input_getter : callable
        Active input extraction function.
    target_getter : callable or None
        Active target extraction function.
    return_format : str
        Selected item representation.
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
        """Return the number of landscape records.

        Returns
        -------
        int
            Number of records in the dataset.
        """
        return len(self.records)

    def __getitem__(self, idx: int):
        """Extract model inputs and optional targets for one record.

        Parameters
        ----------
        idx : int
            Zero-based record index.

        Returns
        -------
        Any
            Inputs alone, an ``(inputs, targets)`` tuple, or a dictionary as
            selected by ``return_format``.
        """
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
    """Manage Landscapy-derived record datasets for Lightning.

    Parameters
    ----------
    train_data : Any
        Non-empty batched records or iterable of training records.
    val_data : Any, optional
        Validation records.
    test_data : Any, optional
        Test records.
    predict_data : Any, optional
        Prediction-only records.
    batch_size : int, default=32
        DataLoader batch size.
    num_workers : int, default=0
        DataLoader worker-process count.
    pin_memory : bool, default=False
        Enable pinned host memory in DataLoaders.
    shuffle : bool, default=True
        Shuffle the training loader.
    val_split : float, default=0.0
        Fraction of training records moved to validation when no validation
        data are supplied.
    val_seed : int or None, optional
        Seed for the optional validation split.
    dataset_factory : callable, default=LandscapeDataset
        Factory constructing datasets from normalized records.
    dataset_kwargs : mapping or None, optional
        Keyword arguments for fit and test datasets.
    predict_dataset_kwargs : mapping or None, optional
        Keyword arguments for prediction datasets. Defaults to
        ``dataset_kwargs``.
    collate_fn : callable or None, optional
        Optional DataLoader collation function.

    Attributes
    ----------
    train_records, val_records, test_records, predict_records : list of dict
        Materialized records for each stage.
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
        self.test_records = (
            normalize_records(test_data) if test_data is not None else []
        )
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
        """Construct datasets required for a Lightning stage.

        Parameters
        ----------
        stage : {"fit", "test", "predict"} or None, optional
            Stage to initialize. ``None`` initializes every stage.

        Returns
        -------
        None
            Datasets are stored on the data module.

        Notes
        -----
        The optional validation split mutates the module's in-memory training
        and validation record lists once during fit setup.
        """
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
        """Return the initialized, optionally shuffled training loader.

        Returns
        -------
        torch.utils.data.DataLoader
            Training DataLoader.

        Raises
        ------
        RuntimeError
            If fit setup has not initialized the training dataset.
        """
        loader = self._loader(self._train_ds, shuffle=self.shuffle)
        if loader is None:
            raise RuntimeError("Training dataset was not initialized.")
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return the validation loader.

        Returns
        -------
        torch.utils.data.DataLoader or None
            Validation loader, or ``None`` when no validation data exist.
        """
        return self._loader(self._val_ds)

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return the test loader.

        Returns
        -------
        torch.utils.data.DataLoader or None
            Test loader, or ``None`` when no test data exist.
        """
        return self._loader(self._test_ds)

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Return the prediction loader.

        Returns
        -------
        torch.utils.data.DataLoader or None
            Prediction loader, or ``None`` when no prediction data exist.
        """
        return self._loader(self._predict_ds)


def _collate_single_graph(items: list[Any]) -> Any:
    return items[0]


def _graph_record_getter(record: Mapping[str, Any]) -> Any:
    if "graph" not in record:
        raise ValueError("Graph dataset records must contain a 'graph' item.")
    return record["graph"]


class LandscapeGraphDataset(LandscapeDataset):
    """Represent one full landscape graph as a single dataset item.

    Parameters
    ----------
    records : sequence of mappings
        Non-empty records containing a ``graph`` entry.
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
    """Build a PyG-style regression graph from a fitness landscape.

    Known numeric fitness values become supervised targets. Unknown values
    should be encoded as ``NaN`` in the source layer. Predefined
    train/validation/test indices can be supplied; otherwise finite targets are
    randomly split using ``val_fraction`` and ``test_fraction``.

    Parameters
    ----------
    landscape : Any
        Landscapy object exposing a graph tensor export or graph and sequences.
    target_layer : str
        Name of the numeric fitness layer used as ``graph.y``.
    tokenizer : Any, str, or None, optional
        Tokenizer forwarded to ``landscape.to_graph_tensor``.
    aggregate_func : callable or None, default=numpy.mean
        Replicate aggregation function for the numeric target layer.
    val_fraction : float, default=0.0
        Fraction of finite targets sampled for validation.
    test_fraction : float, default=0.0
        Fraction of finite targets sampled for testing.
    train_indices : sequence of int, torch.Tensor, or None, optional
        Explicit training node indices.
    val_indices : sequence of int, torch.Tensor, or None, optional
        Explicit validation node indices.
    test_indices : sequence of int, torch.Tensor, or None, optional
        Explicit test node indices.
    seed : int or None, optional
        Random split seed.
    normalize_features : bool, default=False
        Attach training-row feature mean and scale tensors to the graph.
    feature_normalization_eps : float, default=1e-8
        Minimum feature scale before replacement by one.

    Returns
    -------
    Any
        PyTorch Geometric-style graph carrying ``x``, ``edge_index``, ``y``,
        known/train/validation/test/prediction masks, and optional feature
        normalization tensors. Targets and generated features are ``float32``.

    Raises
    ------
    ImportError
        If fallback graph conversion requires unavailable PyTorch Geometric.
    ValueError
        If the landscape, target layer, split fractions, targets, or indices
        violate the graph-regression contract.

    Notes
    -----
    Explicit split indices are node positions in the landscape's canonical
    sequence order. Graph construction and split choice remain scientific
    inputs supplied by the caller.
    """
    if not hasattr(landscape, "to_graph_tensor") and not hasattr(landscape, "graph"):
        raise ValueError(
            "Landscape must implement to_graph_tensor() or expose a graph."
        )
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
        raise ValueError(
            "Landscape graph tensor fallback requires graph and sequences."
        )

    node_order = list(getattr(landscape, "_node_order", list(source_graph.nodes())))
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    embedding = (
        landscape.get_embedding() if hasattr(landscape, "get_embedding") else None
    )
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
    """Manage a single full graph for node-level landscape regression.

    Parameters
    ----------
    train_graph : Any
        Graph containing training masks and targets.
    val_graph : Any, optional
        Validation graph, commonly the same object as ``train_graph``.
    test_graph : Any, optional
        Test graph, commonly the same object as ``train_graph``.
    predict_graph : Any, optional
        Graph used for prediction.
    num_workers : int, default=0
        DataLoader worker-process count.
    pin_memory : bool, default=False
        Enable pinned host memory.

    Attributes
    ----------
    train_graph, val_graph, test_graph, predict_graph : Any
        Full-graph objects assigned to each Lightning stage.
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
        """Build a regression graph and wrap it in a data module.

        Parameters
        ----------
        landscape : Any
            Landscapy fitness landscape.
        target_layer : str
            Numeric fitness layer used as the supervised target.
        tokenizer : Any, str, or None, optional
            Tokenizer forwarded to graph tensor export.
        aggregate_func : callable or None, default=numpy.mean
            Replicate aggregation function.
        val_fraction : float, default=0.0
            Fraction of finite targets sampled for validation.
        test_fraction : float, default=0.0
            Fraction of finite targets sampled for testing.
        train_indices : sequence of int, torch.Tensor, or None, optional
            Explicit training node indices.
        val_indices : sequence of int, torch.Tensor, or None, optional
            Explicit validation node indices.
        test_indices : sequence of int, torch.Tensor, or None, optional
            Explicit test node indices.
        seed : int or None, optional
            Random split seed.
        normalize_features : bool, default=False
            Compute feature statistics from training nodes.
        feature_normalization_eps : float, default=1e-8
            Minimum non-unit feature scale.
        num_workers : int, default=0
            DataLoader worker-process count.
        pin_memory : bool, default=False
            Enable pinned host memory.

        Returns
        -------
        LandscapeGraphRegressionDataModule
            Data module sharing the constructed graph across applicable
            training, validation, test, and prediction stages.
        """
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
