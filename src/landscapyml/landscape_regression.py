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

from .core.inference import infer_fitness_layer_from_landscape
from .core.trainer import TrainingJob, _MODEL_REGISTRY


@dataclass(frozen=True)
class SplitIndices:
    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)

    def as_mapping(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class LandscapeRegressionConfig:
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
    if name in _LANDSCAPE_REGRESSION_RUNNERS and not overwrite:
        raise ValueError(f"Landscape regression runner {name!r} is already registered.")
    _LANDSCAPE_REGRESSION_RUNNERS[name] = runner


def available_landscape_regression_runners() -> list[str]:
    return sorted(_LANDSCAPE_REGRESSION_RUNNERS)


def import_builtin_examples() -> None:
    """
    Import bundled examples for their registry side effects.

    Optional dependencies remain lazy inside the examples where possible.
    """

    for module_name in (
        "landscapyml.examples.gat_fitness",
        "landscapyml.examples.gp_fitness",
    ):
        importlib.import_module(module_name)


def discover_demo_csvs(demo_root: Path) -> list[Path]:
    patterns = ("*.csv", "*.csv.gz")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(demo_root.glob(f"*/*/{pattern}"))
    return sorted(path for path in paths if path.is_file())


def run_landscape_regression(
    config: LandscapeRegressionConfig,
) -> list[dict[str, Any]]:
    import_builtin_examples()
    csv_paths = [Path(path) for path in config.csv_paths]
    if config.demo_root is not None:
        csv_paths.extend(discover_demo_csvs(Path(config.demo_root)))
    csv_paths = sorted(dict.fromkeys(path.resolve() for path in csv_paths))
    if not csv_paths:
        raise ValueError("No CSV inputs were supplied or discovered.")

    results: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        output_path = _output_path_for_csv(csv_path, config.model_key, config.output_suffix)
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
    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        has_split = split_column in reader.fieldnames
        has_validation = validation_column in reader.fieldnames
        for idx, row in enumerate(reader):
            total = idx + 1
            is_validation = has_validation and _parse_bool(row.get(validation_column))
            split_value = str(row.get(split_column, "")).strip().lower() if has_split else ""
            if is_validation:
                val.append(idx)
            elif not has_split or split_value == train_label_norm:
                train.append(idx)
            elif split_value == test_label_norm:
                test.append(idx)
    if not train:
        remaining = sorted(set(range(total)) - set(val) - set(test))
        train.extend(remaining)
    if not train:
        raise ValueError(f"No training rows could be inferred from {path}.")
    return SplitIndices(train=train, val=val, test=test)


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


def _component_summary(graph: nx.Graph) -> dict[str, Any]:
    undirected = graph.to_undirected() if graph.is_directed() else graph
    sizes = sorted((len(component) for component in nx.connected_components(undirected)), reverse=True)
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
        "pearson": _correlation(y, pred),
        "spearman": _correlation(_rankdata(y), _rankdata(pred)),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


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
