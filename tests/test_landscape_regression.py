import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

import landscapyml.landscape_regression as regression
from landscapyml.landscape_regression import _read_split_indices, _regression_metrics

def test_read_split_indices_rejects_unknown_labels_with_row_numbers(tmp_path):
    path = _write_csv(
        tmp_path / "unknown.csv",
        "sequence,set,validation\nAAA,train,false\nBBB,holdout,false\n",
    )

    with pytest.raises(ValueError, match=r"'holdout' \(rows 3\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


def test_read_split_indices_rejects_blank_labels(tmp_path):
    path = _write_csv(
        tmp_path / "blank.csv",
        "sequence,set,validation\nAAA,train,false\nBBB,,false\n",
    )

    with pytest.raises(ValueError, match=r"'<blank>' \(rows 3\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


def test_read_split_indices_normalizes_case_and_whitespace(tmp_path):
    path = _write_csv(
        tmp_path / "normalized.csv",
        'sequence,set,validation\nAAA," Train ",false\nBBB,TEST,false\n',
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="TRAIN",
        test_label="test",
    )

    assert splits.train == [0]
    assert splits.val == []
    assert splits.test == [1]
    assert splits.counts() == {"train": 1, "val": 0, "test": 1, "total": 2}


def test_read_split_indices_gives_validation_precedence(tmp_path):
    path = _write_csv(
        tmp_path / "validation.csv",
        "sequence,set,validation\nAAA,unknown,true\nBBB,train,false\n",
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits.train == [1]
    assert splits.val == [0]
    assert splits.test == []


def test_read_split_indices_assigns_all_rows_when_split_column_is_absent(tmp_path):
    path = _write_csv(
        tmp_path / "no_split.csv",
        "sequence,validation\nAAA,false\nBBB,true\nCCC,false\n",
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits.train == [0, 2]
    assert splits.val == [1]
    assert splits.test == []
    assert splits.counts()["total"] == 3


def test_read_split_indices_rejects_empty_files(tmp_path):
    path = _write_csv(tmp_path / "empty.csv", "sequence,set,validation\n")

    with pytest.raises(ValueError, match=r"No training rows.*\(0 data rows\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


def test_regression_metrics_preserve_average_tie_spearman_definition():
    metrics = _regression_metrics(
        np.asarray([1.0, 1.0, 2.0, 3.0]),
        np.asarray([3.0, 1.0, 1.0, 2.0]),
        [0, 1, 2, 3],
    )

    assert metrics["pearson"] == pytest.approx(-0.0909090909090909)
    assert metrics["spearman"] == pytest.approx(-0.05555555555555556)


@pytest.mark.parametrize(
    ("target", "prediction"),
    [([1.0], [2.0]), ([1.0, 1.0], [1.0, 2.0]), ([1.0, 2.0], [3.0, 3.0])],
)
def test_regression_metrics_keep_undefined_correlations_as_none(target, prediction):
    metrics = _regression_metrics(
        np.asarray(target),
        np.asarray(prediction),
        list(range(len(target))),
    )

    assert metrics["pearson"] is None
    assert metrics["spearman"] is None
@pytest.fixture(autouse=True)
def restore_runner_registry():
    original = dict(regression._LANDSCAPE_REGRESSION_RUNNERS)
    yield
    regression._LANDSCAPE_REGRESSION_RUNNERS.clear()
    regression._LANDSCAPE_REGRESSION_RUNNERS.update(original)


def _write_csv(path: Path, rows: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt", newline="") as handle:
            handle.write(rows)
    else:
        path.write_text(rows, encoding="utf-8")
    return path


def test_split_indices_accessors_return_copies() -> None:
    splits = regression.SplitIndices(train=[0, 1], val=[2], test=[3])

    mapping = splits.as_mapping()
    mapping["train"].append(9)
    assert splits.train == [0, 1]
    assert splits.counts() == {"train": 2, "val": 1, "test": 1, "total": 4}


def test_runner_registration_duplicate_and_overwrite() -> None:
    first = lambda context: {"runner": "first"}
    second = lambda context: {"runner": "second"}
    regression.register_landscape_regression_runner("unit", first, overwrite=True)

    with pytest.raises(ValueError, match="already registered"):
        regression.register_landscape_regression_runner("unit", second)

    regression.register_landscape_regression_runner("unit", second, overwrite=True)
    assert "unit" in regression.available_landscape_regression_runners()


def test_import_builtin_examples_imports_both_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []
    monkeypatch.setattr(regression.importlib, "import_module", imported.append)

    regression.import_builtin_examples()

    assert imported == [
        "landscapyml.examples.gat_fitness",
        "landscapyml.examples.gp_fitness",
    ]


def test_discover_demo_csvs_finds_csv_and_gzip_only(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "dataset" / "split" / "a.csv", "x\n1\n")
    gz_path = _write_csv(tmp_path / "dataset" / "split" / "b.csv.gz", "x\n2\n")
    _write_csv(tmp_path / "dataset" / "split" / "ignored.txt", "x\n3\n")
    _write_csv(tmp_path / "too-shallow.csv", "x\n4\n")

    assert regression.discover_demo_csvs(tmp_path) == [csv_path, gz_path]


def test_read_split_indices_supports_gzip_and_validation_precedence(
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "splits.csv.gz",
        "sequence,set,validation\nAAA,TRAIN,yes\nAAC,train,no\nAAG,TEST,0\n",
    )

    splits = regression._read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits == regression.SplitIndices(train=[1], val=[0], test=[2])


def test_read_split_indices_defaults_rows_to_train_without_split_column(
    tmp_path: Path,
) -> None:
    path = _write_csv(tmp_path / "plain.csv", "sequence,target\nAAA,1\nAAC,2\n")

    splits = regression._read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits.train == [0, 1]
    assert splits.val == []
    assert splits.test == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("", "does not contain a CSV header"),
        ("sequence,set\nAAA,test\n", "No training rows could be inferred"),
    ],
)
def test_read_split_indices_reports_invalid_split_files(
    tmp_path: Path,
    rows: str,
    message: str,
) -> None:
    path = _write_csv(tmp_path / "invalid.csv", rows)

    with pytest.raises(ValueError, match=message):
        regression._read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


@pytest.mark.parametrize("truthy", [1, "TRUE", " t ", "Yes", "y"])
def test_parse_bool_truthy_values(truthy) -> None:
    assert regression._parse_bool(truthy) is True


def test_output_path_normalizes_extensions_and_model_name(tmp_path: Path) -> None:
    path = tmp_path / "data.csv.gz"

    output = regression._output_path_for_csv(path, "org/model:v1", ".metrics")

    assert output.name == "data.org_model_v1.metrics.json"


def test_run_landscape_regression_requires_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(regression, "import_builtin_examples", lambda: None)

    with pytest.raises(ValueError, match="No CSV inputs"):
        regression.run_landscape_regression(
            regression.LandscapeRegressionConfig(model_key="unit")
        )


def test_run_landscape_regression_deduplicates_and_continues_after_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _write_csv(tmp_path / "a.csv", "sequence,target\nAAA,1\n")
    second = _write_csv(tmp_path / "b.csv", "sequence,target\nAAC,2\n")
    calls = []
    monkeypatch.setattr(regression, "import_builtin_examples", lambda: None)

    def fake_run(config, csv_path, output_path):
        calls.append(csv_path)
        if csv_path == first.resolve():
            raise RuntimeError("bad input")
        return {"status": "ok", "output_path": str(output_path)}

    monkeypatch.setattr(regression, "run_landscape_regression_csv", fake_run)
    config = regression.LandscapeRegressionConfig(
        model_key="unit",
        csv_paths=[first, first, second],
        continue_on_error=True,
    )

    results = regression.run_landscape_regression(config)

    assert calls == [first.resolve(), second.resolve()]
    assert [result["status"] for result in results] == ["error", "ok"]
    error_path = Path(results[0]["output_path"])
    assert json.loads(error_path.read_text(encoding="utf-8"))["error"] == (
        "RuntimeError: bad input"
    )


def test_run_landscape_regression_stops_on_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_csv(tmp_path / "a.csv", "sequence,target\nAAA,1\n")
    monkeypatch.setattr(regression, "import_builtin_examples", lambda: None)
    monkeypatch.setattr(
        regression,
        "run_landscape_regression_csv",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    with pytest.raises(RuntimeError, match="stop"):
        regression.run_landscape_regression(
            regression.LandscapeRegressionConfig(model_key="unit", csv_paths=[path])
        )


def test_run_csv_dispatches_registered_runner_and_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_csv(
        tmp_path / "input.csv",
        "sequence,target,set\nAAA,1,train\nAAC,2,test\n",
    )
    layer = SimpleNamespace(to_scalar=lambda: np.array([1.0, 2.0]))
    landscape = SimpleNamespace(fitness_layers={"target": layer})
    monkeypatch.setattr(
        regression,
        "_build_connected_landscape",
        lambda *args, **kwargs: (landscape, {"used": "hamming"}),
    )

    def runner(context):
        assert context.splits == regression.SplitIndices(train=[0], test=[1])
        return {"status": "ok", "values": np.array([1, 2])}

    regression.register_landscape_regression_runner("unit", runner, overwrite=True)
    config = regression.LandscapeRegressionConfig(model_key="unit", csv_paths=[path])

    result = regression.run_landscape_regression_csv(config, path)

    assert result["status"] == "ok"
    output = tmp_path / "input.unit.results.json"
    assert json.loads(output.read_text(encoding="utf-8"))["values"] == [1, 2]


def test_graph_strategy_provenance_for_connected_and_disconnected_inputs(
    tmp_path: Path,
) -> None:
    connected = _write_csv(
        tmp_path / "connected.csv",
        "sequence,target\nAAA,0\nAAC,1\n",
    )
    disconnected = _write_csv(
        tmp_path / "disconnected.csv",
        "sequence,target\nAAA,0\nCCC,1\n",
    )

    _, hamming_info = regression._build_connected_landscape(
        connected,
        sequence_column="sequence",
        target_column="target",
        moltype=None,
    )
    _, knn_info = regression._build_connected_landscape(
        disconnected,
        sequence_column="sequence",
        target_column="target",
        moltype=None,
    )

    assert hamming_info["used"] == "hamming"
    assert hamming_info["hamming"]["component_count"] == 1
    assert knn_info["used"] == "knn"
    assert knn_info["hamming"]["component_count"] == 2
    assert knn_info["knn"]["component_count"] == 1


def test_variable_length_composition_helper_and_provenance_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fitness_landscape._const import PROT_20
    from fitness_landscape.core import landscape as landscape_module

    path = _write_csv(
        tmp_path / "variable.csv",
        "sequence,target\nAAA,0\nAAAA,1\n",
    )
    fallback_landscape, k = regression._build_composition_knn_landscape_from_csv(
        path,
        sequence_column="sequence",
        target_column="target",
        moltype=None,
        alphabet=PROT_20,
    )
    monkeypatch.setattr(
        landscape_module,
        "read_csv_landscape",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("sequences must all have the same length")
        ),
    )
    monkeypatch.setattr(
        regression,
        "_build_composition_knn_landscape_from_csv",
        lambda *args, **kwargs: (fallback_landscape, k),
    )

    landscape, info = regression._build_connected_landscape(
        path,
        sequence_column="sequence",
        target_column="target",
        moltype=None,
    )

    assert landscape is fallback_landscape
    assert info["used"] == "composition_knn"
    assert info["composition_knn"]["component_count"] == 1


@pytest.mark.parametrize("missing_column", ["sequence", "target"])
def test_composition_helper_validates_columns(
    tmp_path: Path,
    missing_column: str,
) -> None:
    from fitness_landscape._const import PROT_20

    columns = [column for column in ("sequence", "target") if column != missing_column]
    path = _write_csv(tmp_path / "missing.csv", f"{columns[0]}\nAAA\n")

    with pytest.raises(ValueError, match=f"missing {missing_column} column"):
        regression._build_composition_knn_landscape_from_csv(
            path,
            sequence_column="sequence",
            target_column="target",
            moltype=None,
            alphabet=PROT_20,
        )


def test_composition_graph_and_component_summary_edge_cases() -> None:
    graph = regression._composition_knn_graph(["AAA"], k=10, alphabet=list("AC"))
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0

    directed = nx.DiGraph()
    directed.add_nodes_from(range(25))
    summary = regression._component_summary(directed)
    assert summary["component_count"] == 25
    assert len(summary["component_sizes"]) == 20
    assert summary["component_sizes_truncated"] is True


def test_unknown_registered_model_lists_available_runners() -> None:
    context = regression.LandscapeRegressionContext(
        config=regression.LandscapeRegressionConfig(model_key="missing"),
        csv_path=Path("input.csv"),
        output_path=Path("output.json"),
        landscape=object(),
        graph_info={},
        splits=regression.SplitIndices(),
        target_values=np.array([]),
    )

    with pytest.raises(ValueError, match="Unknown model_key"):
        regression._run_registered_lightning_model(context)


def test_metrics_and_json_serialization_helpers() -> None:
    target = np.array([1.0, 2.0, np.nan])
    prediction = np.array([1.5, 1.5, 3.0])
    splits = regression.SplitIndices(train=[0, 1], val=[2])

    metrics = regression._split_metrics(target, prediction, splits)
    assert metrics["train"]["rmse"] == 0.5
    assert metrics["val"]["n"] == 0
    assert metrics["test"]["pearson"] is None
    assert (
        regression._correlation(
            np.array([1.0]), np.array([1.0]), method="pearson"
        )
        is None
    )
    assert (
        regression._correlation(np.ones(2), np.arange(2), method="spearman")
        is None
    )
    assert regression._to_jsonable(torch.tensor([1.0, 2.0])) == [1.0, 2.0]
    assert regression._to_jsonable(np.float64(3.0)) == 3.0
