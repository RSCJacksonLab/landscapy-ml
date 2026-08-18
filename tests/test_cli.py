from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

cli_module = importlib.import_module("landscapyml.__main__")


def test_list_command_reports_all_registry_sections() -> None:
    result = CliRunner().invoke(cli_module.cli, ["list"])

    assert result.exit_code == 0
    assert "Models:" in result.output
    assert "graph_attention_regressor" in result.output
    assert "Data builders:" in result.output
    assert "Landscape regression runners:" in result.output
    assert "diffusion_prior_gp" in result.output


@pytest.mark.parametrize(
    "option",
    ["--model-kwargs", "--data-kwargs", "--trainer-kwargs", "--fit-kwargs"],
)
def test_train_command_rejects_non_object_json(tmp_path: Path, option: str) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("sequence,target\nAAA,1\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "train-landscape",
            "--csv-path",
            str(csv_path),
            "--model-key",
            "test",
            option,
            "[]",
        ],
    )

    assert result.exit_code == 1
    assert f"{option} must decode to a JSON object" in result.output


def test_train_command_rejects_malformed_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("sequence,target\nAAA,1\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "train-landscape",
            "--csv-path",
            str(csv_path),
            "--model-key",
            "test",
            "--model-kwargs",
            "{bad",
        ],
    )

    assert result.exit_code == 1
    assert "--model-kwargs must be a valid JSON object" in result.output


@pytest.mark.parametrize(
    ("devices", "expected_devices"),
    [("2", 2), ("auto", "auto")],
)
def test_train_command_builds_config_and_reports_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    devices: str,
    expected_devices: int | str,
) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("sequence,target\nAAA,1\n", encoding="utf-8")
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return [{"status": "ok", "output_path": tmp_path / "result.json"}]

    monkeypatch.setattr(cli_module, "run_landscape_regression", fake_run)
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "train-landscape",
            "--csv-path",
            str(csv_path),
            "--model-key",
            "test",
            "--devices",
            devices,
            "--continue-on-error",
            "--model-kwargs",
            '{"hidden": 4}',
            "--trainer-kwargs",
            "",
        ],
    )

    assert result.exit_code == 0
    assert "ok:" in result.output
    config = captured["config"]
    assert config.devices == expected_devices
    assert config.continue_on_error is True
    assert config.model_kwargs == {"hidden": 4}
    assert config.trainer_kwargs == {}


def test_cli_usage_errors_have_nonzero_exit_code() -> None:
    result = CliRunner().invoke(cli_module.cli, ["train-landscape"])

    assert result.exit_code == 2
    assert "Missing option '--model-key'" in result.output


def test_main_returns_zero_and_propagates_system_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module.cli, "main", lambda **kwargs: None)
    assert cli_module.main(["list"]) == 0

    def exit_three(**kwargs):
        raise SystemExit(3)

    monkeypatch.setattr(cli_module.cli, "main", exit_three)
    assert cli_module.main(["list"]) == 3
