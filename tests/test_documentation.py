from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _fenced_blocks(path: Path, language: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    return re.findall(pattern, text, flags=re.DOTALL)


def test_python_documentation_blocks_compile() -> None:
    documentation = [REPOSITORY_ROOT / "README.md"]
    documentation.extend(sorted((REPOSITORY_ROOT / "docs").glob("*.md")))

    checked = 0
    for path in documentation:
        for index, block in enumerate(_fenced_blocks(path, "python"), start=1):
            compile(block, f"{path.name}:python-block-{index}", "exec")
            checked += 1

    assert checked > 0


def test_readme_python_example_uses_current_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoOpTrainer:
        def fit(self, model, *, datamodule) -> None:
            assert model is not None
            assert datamodule.train_graph.num_nodes == 8

    import landscapyml.core.trainer

    monkeypatch.setattr(
        landscapyml.core.trainer,
        "create_trainer",
        lambda **kwargs: NoOpTrainer(),
    )
    blocks = _fenced_blocks(REPOSITORY_ROOT / "README.md", "python")
    assert len(blocks) == 1

    namespace: dict[str, object] = {}
    exec(compile(blocks[0], "README.md", "exec"), namespace)

    landscape = namespace["landscape"]
    assert "gat_predicted_fitness" in landscape.fitness_layers


@pytest.mark.parametrize(
    ("script_name", "expected_args"),
    [
        (
            "run_gat.sh",
            [
                "--model-key",
                "graph_attention_regressor",
                "--max-epochs",
                "1",
            ],
        ),
        (
            "run_gp.sh",
            [
                "--model-key",
                "diffusion_prior_gp",
                "--fit-kwargs",
                '{"training_iters": 2, "learning_rate": 0.05}',
            ],
        ),
    ],
)
def test_demo_scripts_use_repository_environment_and_build_command(
    tmp_path: Path,
    script_name: str,
    expected_args: list[str],
) -> None:
    repository = tmp_path / "checkout"
    demo_dir = repository / "demo"
    demo_dir.mkdir(parents=True)
    script = demo_dir / script_name
    shutil.copy2(REPOSITORY_ROOT / "demo" / script_name, script)

    capture_path = tmp_path / f"{script_name}.args"
    python_bin = repository / ".env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${CAPTURE_PATH}"\n',
        encoding="utf-8",
    )
    python_bin.chmod(0o755)

    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "sequence,target,set,validation\nAAA,0.0,train,false\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("PYTHON", None)
    environment.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "FIT_KWARGS": '{"training_iters": 2, "learning_rate": 0.05}',
            "MAX_EPOCHS": "1",
        }
    )

    subprocess.run(
        ["bash", str(script), "--csv-path", str(csv_path)],
        check=True,
        cwd=repository,
        env=environment,
    )

    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["-m", "landscapyml", "train-landscape"]
    assert ["--csv-path", str(csv_path)] == args[-2:]
    for expected in expected_args:
        assert expected in args
