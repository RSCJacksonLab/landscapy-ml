from __future__ import annotations

import re
import tomllib
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


def test_readme_python_example_uses_portable_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    blocks = _fenced_blocks(REPOSITORY_ROOT / "README.md", "python")
    assert len(blocks) == 1

    namespace: dict[str, object] = {}
    exec(compile(blocks[0], "README.md", "exec"), namespace)

    dataset = namespace["dataset"]
    landscape = namespace["landscape"]
    features, target = dataset[0]
    assert len(dataset) == 4
    assert features.shape == (3, 2)
    assert target.shape == ()
    assert list(landscape.fitness_layers) == ["target"]


def test_documentation_uses_portable_fixture_and_no_semicolons() -> None:
    fixture = REPOSITORY_ROOT / "src/landscapyml/data/minimal_landscape.csv"
    assert fixture.is_file()

    documentation = [REPOSITORY_ROOT / "README.md"]
    documentation.extend(sorted((REPOSITORY_ROOT / "docs").glob("*.md")))
    for path in documentation:
        text = path.read_text(encoding="utf-8")
        assert ";" not in text, (
            f"Semicolon found in {path.relative_to(REPOSITORY_ROOT)}"
        )

    relative_fixture = fixture.relative_to(REPOSITORY_ROOT).as_posix()
    assert "data/minimal_landscape.csv" in (REPOSITORY_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert relative_fixture in (REPOSITORY_ROOT / "docs/cli.md").read_text(
        encoding="utf-8"
    )


def test_default_install_contains_every_non_dev_extra() -> None:
    metadata = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    default_requirements = set(metadata["dependencies"])

    for name, requirements in metadata["optional-dependencies"].items():
        if name == "dev":
            continue
        assert set(requirements) <= default_requirements, (
            f"Optional dependency group {name!r} is not included in the default install"
        )
