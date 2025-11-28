import json
import sys
import types
from pathlib import Path

from landscapyml.data_utils import (
    CSVConfigRequest,
    build_config_from_csv,
    build_config_from_dataframe,
    write_config,
)


class FakeCat:
    def __init__(self, labels):
        cats = list(dict.fromkeys(labels))
        code_list = [cats.index(lbl) for lbl in labels]

        class Codes:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        class Categories:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        self._codes = Codes(code_list)
        self._categories = Categories(cats)

    @property
    def codes(self):
        return self._codes

    @property
    def categories(self):
        return self._categories


class FakeSeries:
    def __init__(self, values):
        self.values = list(values)

    def astype(self, dtype):
        assert dtype == "category"
        return self

    @property
    def cat(self):
        return FakeCat(self.values)

    def tolist(self):
        return list(self.values)


class FakeColumn:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeDataFrame:
    def __init__(self, sequences, labels):
        self._data = {
            "sequence": FakeColumn(sequences),
            "family": FakeSeries(labels),
        }

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


def test_build_config_from_dataframe_creates_expected_fields():
    df = FakeDataFrame(["AAA", "BBB"], ["x", "y"])
    cfg = build_config_from_dataframe(
        df,
        sequence_column="sequence",
        label_column="family",
        max_epochs=3,
        model="sequence_gp_classifier",
        embedding_mode="hard",
        val_split=0.25,
        wandb_project="proj",
    )
    assert cfg["model"] == "sequence_gp_classifier"
    assert cfg["data_kwargs"]["label_key"] == "family"
    assert cfg["data_kwargs"]["label_mapping"] == ["x", "y"]
    assert cfg["trainer_kwargs"]["max_epochs"] == 3
    assert cfg["trainer_kwargs"]["wandb_project"] == "proj"


def test_build_config_from_csv_uses_patched_pandas(monkeypatch):
    df = FakeDataFrame(["AAA"], ["z"])

    def fake_require():
        return None

    fake_pd = types.SimpleNamespace(read_csv=lambda path: df)
    monkeypatch.setattr("landscapyml.data_utils._require_pandas", lambda: None)
    monkeypatch.setitem(sys.modules, "pandas", fake_pd)

    cfg = build_config_from_csv(
        CSVConfigRequest(
            csv_path=Path("dummy.csv"),
            sequence_column="sequence",
            label_column="family",
            max_epochs=2,
            use_wandb=False,
            embedding_mode="hard",
            model_key="sequence_mlp_classifier",
        )
    )
    assert cfg["model"] == "sequence_mlp_classifier"
    assert cfg["trainer_kwargs"]["max_epochs"] == 2
    assert cfg["data_kwargs"]["train_sequences"] == ["AAA"]


def test_write_config_writes_json(tmp_path):
    cfg = {"a": 1, "b": {"c": 2}}
    out_path = tmp_path / "config.json"
    write_config(cfg, out_path)
    loaded = json.loads(out_path.read_text())
    assert loaded == cfg
