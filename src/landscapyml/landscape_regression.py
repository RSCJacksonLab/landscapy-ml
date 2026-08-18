"""End-to-end CSV workflows for landscape regression examples."""

from __future__ import annotations

import csv
import gzip
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .core.data_utils import sequence_composition_features
from .core.inference import infer_fitness_layer_from_landscape
from .core.model_registry import _MODEL_REGISTRY
from .core.trainer import TrainingJob


@dataclass(frozen=True)
class SplitIndices:
    """Store canonical node positions for supervised data splits.

    Parameters
    ----------
    train : list of int, optional
        Training node positions.
    val : list of int, optional
        Validation node positions.
    test : list of int, optional
        Test node positions.

    Attributes
    ----------
    train, val, test : list of int
        Node positions for each split.
    """

    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)

    def as_mapping(self) -> dict[str, list[int]]:
        """Return defensive list copies keyed by split name.

        Returns
        -------
        dict of str to list of int
            ``train``, ``val``, and ``test`` node positions.
        """
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def counts(self) -> dict[str, int]:
        """Return the number of rows assigned to each split.

        Returns
        -------
        dict of str to int
            Counts for ``train``, ``val``, ``test``, and their ``total``.
        """
        counts = {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }
        counts["total"] = sum(counts.values())
        return counts


@dataclass(frozen=True)
class LandscapeRegressionConfig:
    """Configure CSV landscape construction, fitting, and output.

    Parameters
    ----------
    model_key : str
        Registered Lightning model or custom landscape-runner key.
    csv_paths : sequence of pathlib.Path, optional
        Explicit CSV or compressed CSV inputs.
    demo_root : pathlib.Path or None, optional
        Root searched for ``dataset/split/*.csv`` inputs.
    sequence_column : str, default="sequence"
        Input sequence column.
    target_column : str, default="target"
        Numeric fitness column.
    split_column : str, default="set"
        Column containing train and test labels.
    validation_column : str, default="validation"
        Boolean validation indicator column.
    train_label : str, default="train"
        Training value in ``split_column``.
    test_label : str, default="test"
        Test value in ``split_column``.
    data_name : str, default="landscape_graph_regression"
        Registered data-builder key for Lightning models.
    output_suffix : str, default="results"
        Suffix for output JSON paths.
    seed : int or None, optional
        Random seed for splitting and training.
    max_epochs : int, default=50
        Maximum Lightning training epochs.
    accelerator : str or None, default="auto"
        Lightning accelerator selection.
    devices : int, str, or None, default=1
        Lightning device selection.
    model_kwargs : mapping, optional
        Model-factory keyword arguments.
    data_kwargs : mapping, optional
        Data-builder keyword arguments.
    trainer_kwargs : mapping, optional
        Trainer keyword overrides.
    fit_kwargs : mapping, optional
        Custom non-Lightning runner fit arguments.
    tokenizer : Any, str, or None, optional
        Tokenizer passed to graph conversion.
    moltype : str or None, default="protein"
        Landscapy sequence molecular type.
    continue_on_error : bool, default=False
        Record per-file errors and continue rather than raising immediately.

    Notes
    -----
    Graph selection performed by this legacy example runner is recorded in
    each result. It is a modeling choice and should not be interpreted as
    biological validation of the selected representation.
    """

    model_key: str
    csv_paths: Sequence[Path] = field(default_factory=tuple)
    demo_root: Path | None = None
    sequence_column: str = "sequence"
    target_column: str = "target"
    split_column: str = "set"
    validation_column: str = "validation"
    train_label: str = "train"
    test_label: str = "test"
    data_name: str = "landscape_graph_regression"
    output_suffix: str = "results"
    seed: int | None = None
    max_epochs: int = 50
    accelerator: str | None = "auto"
    devices: int | str | None = 1
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    data_kwargs: Mapping[str, Any] = field(default_factory=dict)
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    fit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    tokenizer: Any | str | None = None
    moltype: str | None = "protein"
    continue_on_error: bool = False


@dataclass
class LandscapeRegressionContext:
    """Bundle prepared inputs passed to a landscape regression runner.

    Parameters
    ----------
    config : LandscapeRegressionConfig
        Shared workflow configuration.
    csv_path : pathlib.Path
        Current source dataset.
    output_path : pathlib.Path
        JSON result destination.
    landscape : Any
        Prepared Landscapy fitness landscape.
    graph_info : dict of str to Any
        Recorded graph construction and component summary.
    splits : SplitIndices
        Canonical supervised node positions.
    target_values : numpy.ndarray
        One-dimensional numeric targets aligned with landscape sequence order.
    """

    config: LandscapeRegressionConfig
    csv_path: Path
    output_path: Path
    landscape: Any
    graph_info: dict[str, Any]
    splits: SplitIndices
    target_values: np.ndarray


LandscapeRegressionRunner = Callable[[LandscapeRegressionContext], dict[str, Any]]
_LANDSCAPE_REGRESSION_RUNNERS: dict[str, LandscapeRegressionRunner] = {}


def register_landscape_regression_runner(
    name: str,
    runner: LandscapeRegressionRunner,
    *,
    overwrite: bool = False,
) -> None:
    """Register an end-to-end landscape regression runner.

    Parameters
    ----------
    name : str
        Process-local runner key.
    runner : callable
        Function accepting ``LandscapeRegressionContext`` and returning a
        JSON-serializable result mapping.
    overwrite : bool, default=False
        Replace an existing runner.

    Returns
    -------
    None
        The process-local runner registry is mutated.

    Raises
    ------
    ValueError
        If ``name`` exists and ``overwrite`` is false.
    """
    if name in _LANDSCAPE_REGRESSION_RUNNERS and not overwrite:
        raise ValueError(f"Landscape regression runner {name!r} is already registered.")
    _LANDSCAPE_REGRESSION_RUNNERS[name] = runner


def available_landscape_regression_runners() -> list[str]:
    """Return registered landscape-runner names.

    Returns
    -------
    list of str
        Registry keys in lexical order.
    """
    return sorted(_LANDSCAPE_REGRESSION_RUNNERS)


def import_builtin_examples() -> None:
    """Import bundled examples for their registry side effects.

    Optional dependencies remain lazy inside the examples where possible.

    Returns
    -------
    None
        Bundled example modules are imported.

    Notes
    -----
    Importing registers example models, adapters, data builders, and custom
    landscape runners in process-local registries.
    """
    for module_name in (
        "landscapyml.examples.gat_fitness",
        "landscapyml.examples.gp_fitness",
    ):
        importlib.import_module(module_name)


def discover_demo_csvs(demo_root: Path) -> list[Path]:
    """Discover two-level nested CSV inputs beneath a demo root.

    Parameters
    ----------
    demo_root : pathlib.Path
        Root searched with ``*/*/*.csv`` and ``*/*/*.csv.gz`` patterns.

    Returns
    -------
    list of pathlib.Path
        Existing files in lexical order.
    """
    patterns = ("*.csv", "*.csv.gz")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(demo_root.glob(f"*/*/{pattern}"))
    return sorted(path for path in paths if path.is_file())


def run_landscape_regression(
    config: LandscapeRegressionConfig,
) -> list[dict[str, Any]]:
    """Run one configured regression workflow across selected CSV files.

    Parameters
    ----------
    config : LandscapeRegressionConfig
        Dataset discovery, landscape construction, model, and output settings.

    Returns
    -------
    list of dict
        One result mapping per unique resolved CSV path.

    Raises
    ------
    ValueError
        If no CSV input is supplied or discovered.
    Exception
        Propagates per-file failures unless ``continue_on_error`` is true.

    Notes
    -----
    Each successful or captured-error result is written to JSON beside its
    source CSV. The workflow imports bundled examples for registry side effects.
    """
    import_builtin_examples()
    csv_paths = [Path(path) for path in config.csv_paths]
    if config.demo_root is not None:
        csv_paths.extend(discover_demo_csvs(Path(config.demo_root)))
    csv_paths = sorted(dict.fromkeys(path.resolve() for path in csv_paths))
    if not csv_paths:
        raise ValueError("No CSV inputs were supplied or discovered.")

    results: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        output_path = _output_path_for_csv(
            csv_path, config.model_key, config.output_suffix
        )
        try:
            result = run_landscape_regression_csv(config, csv_path, output_path)
        except Exception as exc:
            if not config.continue_on_error:
                raise
            result = {
                "status": "error",
                "csv_path": str(csv_path),
                "output_path": str(output_path),
                "model_key": config.model_key,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(output_path, result)
        results.append(result)
    return results


def run_landscape_regression_csv(
    config: LandscapeRegressionConfig,
    csv_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare, fit, evaluate, and record one CSV regression task.

    Parameters
    ----------
    config : LandscapeRegressionConfig
        Workflow settings.
    csv_path : pathlib.Path
        CSV or compressed CSV dataset.
    output_path : pathlib.Path or None, optional
        Explicit result path. A deterministic sibling path is generated when
        omitted.

    Returns
    -------
    dict of str to Any
        Runner result written to ``output_path``.

    Raises
    ------
    ImportError
        If required Landscapy, model, or optional runner dependencies are
        unavailable.
    RuntimeError
        If the selected fallback graph is disconnected.
    ValueError
        If CSV schema, graph construction, splits, or model selection is
        invalid.

    Notes
    -----
    The legacy helper requests a Hamming graph and may substitute a k-nearest
    neighbor graph to force connectivity. The result's ``graph`` field records
    the representation actually used; callers must treat that substitution as
    a scientific modeling choice.
    """
    output_path = output_path or _output_path_for_csv(
        csv_path,
        config.model_key,
        config.output_suffix,
    )
    landscape, graph_info = _build_connected_landscape(
        csv_path,
        sequence_column=config.sequence_column,
        target_column=config.target_column,
        moltype=config.moltype,
    )
    splits = _read_split_indices(
        csv_path,
        split_column=config.split_column,
        validation_column=config.validation_column,
        train_label=config.train_label,
        test_label=config.test_label,
    )
    target_values = _layer_values(landscape, config.target_column)
    context = LandscapeRegressionContext(
        config=config,
        csv_path=csv_path,
        output_path=output_path,
        landscape=landscape,
        graph_info=graph_info,
        splits=splits,
        target_values=target_values,
    )
    runner = _LANDSCAPE_REGRESSION_RUNNERS.get(config.model_key)
    if runner is None:
        runner = _run_registered_lightning_model
    result = runner(context)
    _write_json(output_path, result)
    return result


def _open_csv_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def _read_split_indices(
    path: Path,
    *,
    split_column: str,
    validation_column: str,
    train_label: str,
    test_label: str,
) -> SplitIndices:
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    total = 0
    train_label_norm = train_label.strip().lower()
    test_label_norm = test_label.strip().lower()
    if not train_label_norm or not test_label_norm:
        raise ValueError("train_label and test_label must be non-empty strings.")
    if train_label_norm == test_label_norm:
        raise ValueError("train_label and test_label must be distinct.")
    unexpected: dict[str, list[int]] = {}
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        has_split = split_column in reader.fieldnames
        has_validation = validation_column in reader.fieldnames
        for idx, row in enumerate(reader):
            total = idx + 1
            is_validation = has_validation and _parse_bool(row.get(validation_column))
            split_value = (
                str(row.get(split_column, "")).strip().lower() if has_split else ""
            )
            if is_validation:
                val.append(idx)
            elif not has_split or split_value == train_label_norm:
                train.append(idx)
            elif split_value == test_label_norm:
                test.append(idx)
            else:
                display_value = split_value or "<blank>"
                unexpected.setdefault(display_value, []).append(reader.line_num)
    if unexpected:
        details = "; ".join(
            f"{value!r} (rows {', '.join(str(row) for row in rows[:5])})"
            for value, rows in sorted(unexpected.items())
        )
        raise ValueError(
            f"Unexpected values in split column {split_column!r} of {path}: "
            f"{details}. Expected {train_label_norm!r} or {test_label_norm!r}; "
            f"rows marked in {validation_column!r} take validation precedence."
        )
    if not train:
        raise ValueError(
            f"No training rows could be inferred from {path} ({total} data rows)."
        )
    splits = SplitIndices(train=train, val=val, test=test)
    if splits.counts()["total"] != total:
        raise RuntimeError(
            f"Internal split assignment error for {path}: assigned "
            f"{splits.counts()['total']} of {total} rows."
        )
    return splits


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _build_connected_landscape(
    path: Path,
    *,
    sequence_column: str,
    target_column: str,
    moltype: str | None,
) -> tuple[Any, dict[str, Any]]:
    from fitness_landscape._const import PROT_20
    from fitness_landscape.core.landscape import FitnessLandscape, read_csv_landscape

    try:
        landscape = read_csv_landscape(
            path,
            sequence_col=sequence_column,
            alphabet=PROT_20,
            moltype=moltype,
            graph="hamming",
            numeric_layers=[target_column],
            embedding_domain="ohe",
            attach_embeddings=False,
        )
    except ValueError as exc:
        if "must all have the same length" not in str(exc):
            raise
        landscape, k = _build_composition_knn_landscape_from_csv(
            path,
            sequence_column=sequence_column,
            target_column=target_column,
            moltype=moltype,
            alphabet=PROT_20,
        )
        summary = _component_summary(landscape.graph)
        graph_info = {
            "requested": "hamming",
            "used": "composition_knn",
            "hamming_error": str(exc),
            "composition_knn_k": k,
            "composition_knn": summary,
        }
        if summary["component_count"] != 1:
            raise RuntimeError(
                "Composition KNN fallback did not produce a single connected "
                f"component for {path} with k={k}."
            ) from exc
        return landscape, graph_info

    hamming_summary = _component_summary(landscape.graph)
    graph_info: dict[str, Any] = {
        "requested": "hamming",
        "used": "hamming",
        "hamming": hamming_summary,
    }
    if hamming_summary["component_count"] == 1:
        return landscape, graph_info

    k = max(1, int(math.sqrt(len(landscape.sequences))))
    knn_landscape = FitnessLandscape.build(
        sequences=landscape.sequences,
        graph="knn",
        fitness_layers=landscape.fitness_layers,
        embedding_domain="ohe",
        attach_embeddings=False,
        k=k,
    )
    knn_summary = _component_summary(knn_landscape.graph)
    graph_info.update(
        {
            "used": "knn",
            "knn_k": k,
            "knn": knn_summary,
        }
    )
    if knn_summary["component_count"] != 1:
        raise RuntimeError(
            "KNN fallback did not produce a single connected component "
            f"for {path} with k={k}."
        )
    return knn_landscape, graph_info


def _build_composition_knn_landscape_from_csv(
    path: Path,
    *,
    sequence_column: str,
    target_column: str,
    moltype: str | None,
    alphabet: Sequence[Any],
) -> tuple[Any, int]:
    import pandas as pd
    from fitness_landscape.core.fitness import NumericFitness
    from fitness_landscape.core.landscape import FitnessLandscape
    from fitness_landscape.core.sequence import make_sequence

    df = pd.read_csv(path)
    if sequence_column not in df.columns:
        raise ValueError(f"CSV is missing sequence column {sequence_column!r}.")
    if target_column not in df.columns:
        raise ValueError(f"CSV is missing target column {target_column!r}.")

    sequences = [
        make_sequence(value, alphabet=alphabet, moltype=moltype)
        for value in df[sequence_column].tolist()
    ]
    layer = NumericFitness.from_scalars(
        target_column,
        df[target_column].to_numpy(dtype=float),
    )
    n = len(sequences)
    k = max(1, int(math.sqrt(n)))
    graph = _composition_knn_graph(sequences, k=k, alphabet=alphabet)
    summary = _component_summary(graph)
    while summary["component_count"] != 1 and k < max(1, n - 1):
        k = min(n - 1, max(k + 1, k * 2))
        graph = _composition_knn_graph(sequences, k=k, alphabet=alphabet)
        summary = _component_summary(graph)

    return (
        FitnessLandscape.build(
            sequences=sequences,
            graph=graph,
            fitness_layers={target_column: layer},
            embedding_domain="ohe",
            attach_embeddings=False,
        ),
        k,
    )


def _composition_knn_graph(
    sequences: Sequence[Any],
    *,
    k: int,
    alphabet: Sequence[Any],
) -> nx.Graph:
    from scipy.spatial import cKDTree

    n = len(sequences)
    graph = nx.Graph()
    for idx, sequence in enumerate(sequences):
        graph.add_node(idx, sequence=sequence)
    if n <= 1:
        return graph

    k = min(max(1, int(k)), n - 1)
    features = sequence_composition_features(sequences, alphabet=alphabet)
    tree = cKDTree(features)
    distances, indices = tree.query(features, k=k + 1)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]
        distances = distances[:, None]

    for src in range(n):
        for distance, dst in zip(distances[src], indices[src]):
            dst = int(dst)
            if dst == src:
                continue
            graph.add_edge(src, dst, distance=float(distance))
    return graph


def _component_summary(graph: nx.Graph) -> dict[str, Any]:
    undirected = graph.to_undirected() if graph.is_directed() else graph
    sizes = sorted(
        (len(component) for component in nx.connected_components(undirected)),
        reverse=True,
    )
    count = len(sizes)
    return {
        "component_count": count,
        "largest_component_size": sizes[0] if sizes else 0,
        "smallest_component_size": sizes[-1] if sizes else 0,
        "component_sizes": sizes[:20],
        "component_sizes_truncated": count > 20,
    }


def _run_registered_lightning_model(
    context: LandscapeRegressionContext,
) -> dict[str, Any]:
    config = context.config
    if config.model_key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "none"
        custom = ", ".join(available_landscape_regression_runners()) or "none"
        raise ValueError(
            f"Unknown model_key {config.model_key!r}. Registered Lightning models: "
            f"{available}. Custom landscape runners: {custom}."
        )

    trainer_kwargs = {
        "max_epochs": config.max_epochs,
        "accelerator": config.accelerator,
        "devices": config.devices,
        "use_wandb": False,
        "checkpoint_monitor": None,
        "log_dir": str(context.output_path.parent / "logs"),
        "checkpoint_dir": str(
            context.output_path.parent / "checkpoints" / config.model_key
        ),
        "experiment_name": context.output_path.stem,
    }
    trainer_kwargs.update(dict(config.trainer_kwargs))

    data_kwargs = {
        "landscape": context.landscape,
        "target_layer": config.target_column,
        "tokenizer": config.tokenizer,
    }
    data_kwargs.update(dict(config.data_kwargs))

    job = TrainingJob(
        model_name=config.model_key,
        data_name=config.data_name,
        model_kwargs=dict(config.model_kwargs),
        data_kwargs=data_kwargs,
        split_indices=context.splits.as_mapping(),
        trainer_kwargs=trainer_kwargs,
        seed=config.seed,
    )
    trainer, model, _ = job.run(fit=True, test=True)
    pred_layer = infer_fitness_layer_from_landscape(
        context.landscape,
        model,
        input_adapter="graph_tensor",
        input_adapter_kwargs={"tokenizer": config.tokenizer},
        attach=True,
        inplace=True,
        layer_name=f"{config.model_key}_predicted_{config.target_column}",
    )
    predictions = _layer_to_array(pred_layer)
    return _base_result(context) | {
        "status": "ok",
        "runner": "lightning_training_job",
        "metrics": _split_metrics(context.target_values, predictions, context.splits),
        "trainer_metrics": _jsonable_mapping(getattr(trainer, "callback_metrics", {})),
    }


def _run_diffusion_prior_gp(context: LandscapeRegressionContext) -> dict[str, Any]:
    from .examples.gp_fitness import (
        attach_diffusion_gp_predictions,
        fit_diffusion_prior_gp,
    )

    config = context.config
    fit_result = fit_diffusion_prior_gp(
        context.landscape,
        target_layer=config.target_column,
        train_indices=context.splits.train,
        **dict(config.fit_kwargs),
    )
    pred_layer = attach_diffusion_gp_predictions(
        context.landscape,
        fit_result.model,
        layer_name=f"{config.model_key}_predicted_{config.target_column}",
        attach=True,
        inplace=True,
    )
    predictions = _layer_to_array(pred_layer)
    losses = fit_result.losses
    return _base_result(context) | {
        "status": "ok",
        "runner": "diffusion_prior_gp",
        "metrics": _split_metrics(context.target_values, predictions, context.splits),
        "gp": {
            "t_map": fit_result.artifacts.t_map,
            "signal_variance": fit_result.artifacts.signal_variance,
            "fit_node_count": int(fit_result.artifacts.fit_indices.numel()),
            "train_node_count": int(fit_result.artifacts.train_indices.numel()),
            "loss_count": len(losses),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
        },
    }


register_landscape_regression_runner(
    "diffusion_prior_gp",
    _run_diffusion_prior_gp,
    overwrite=True,
)


def _base_result(context: LandscapeRegressionContext) -> dict[str, Any]:
    return {
        "csv_path": str(context.csv_path),
        "output_path": str(context.output_path),
        "model_key": context.config.model_key,
        "target_column": context.config.target_column,
        "sequence_column": context.config.sequence_column,
        "graph": context.graph_info,
        "splits": context.splits.counts(),
    }


def _layer_values(landscape: Any, layer_name: str) -> np.ndarray:
    layer = landscape.fitness_layers[layer_name]
    return _layer_to_array(layer)


def _layer_to_array(layer: Any) -> np.ndarray:
    try:
        values = layer.to_scalar()
    except TypeError:
        values = layer.to_scalar(aggregate_func=np.mean)
    return np.asarray(values, dtype=float).reshape(-1)


def _split_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    splits: SplitIndices,
) -> dict[str, dict[str, Any]]:
    return {
        "train": _regression_metrics(target, prediction, splits.train),
        "val": _regression_metrics(target, prediction, splits.val),
        "test": _regression_metrics(target, prediction, splits.test),
    }


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    indices: Sequence[int],
) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "pearson": None,
            "spearman": None,
        }
    y = target[idx]
    pred = prediction[idx]
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    pred = pred[finite]
    if y.size == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "pearson": None,
            "spearman": None,
        }
    residual = pred - y
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": _correlation(y, pred, method="pearson"),
        "spearman": _correlation(y, pred, method="spearman"),
    }


def _correlation(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: str,
) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    if method == "pearson":
        result = pearsonr(a, b)
    elif method == "spearman":
        result = spearmanr(a, b)
    else:
        raise ValueError(f"Unknown correlation method {method!r}.")
    statistic = float(result.statistic)
    return statistic if np.isfinite(statistic) else None


def _output_path_for_csv(path: Path, model_key: str, suffix: str) -> Path:
    name = path.name
    for ending in (".gz", ".csv"):
        if name.endswith(ending):
            name = name[: -len(ending)]
    safe_model = model_key.replace("/", "_").replace(":", "_")
    safe_suffix = suffix.strip(".") or "results"
    return path.with_name(f"{name}.{safe_model}.{safe_suffix}.json")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _jsonable_mapping(mapping: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _to_jsonable(value) for key, value in mapping.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 0:
            return value.item()
        return value.tolist()
    return value


__all__ = [
    "LandscapeRegressionConfig",
    "LandscapeRegressionContext",
    "SplitIndices",
    "available_landscape_regression_runners",
    "discover_demo_csvs",
    "import_builtin_examples",
    "register_landscape_regression_runner",
    "run_landscape_regression",
    "run_landscape_regression_csv",
]
