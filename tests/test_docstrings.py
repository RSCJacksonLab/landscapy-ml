"""Enforce NumPy-style documentation for the source-level public API."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from numpydoc.validate import validate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "landscapyml"

# These numpydoc checks require optional prose that is maintained centrally in
# README/docs rather than repeated on every small API member. GL01 conflicts
# with the PEP 257/pydocstyle same-line summary selected for this package.
IGNORED_NUMPYDOC_CODES = {"ES01", "EX01", "GL01", "SA01"}

# Protocol classes expose a synthetic (*args, **kwargs) constructor even when
# their source declares no constructor parameters.
OBJECT_CODE_EXCLUSIONS = {
    "landscapyml.core.adaptor.ModelAdapter": {"PR01"},
}

PUBLIC_MAGIC_METHODS = {"__getitem__", "__len__", "__post_init__"}
EXPORTED_PRIVATE_FUNCTIONS = {"landscapyml.core.data_utils._pad_tokens"}
CLICK_COMMANDS = {"cli", "list_registered", "train_landscape"}


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _public_object_names() -> list[str]:
    names: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{module}.{node.name}"
                if node.name.startswith("_") and name not in EXPORTED_PRIVATE_FUNCTIONS:
                    continue
                if module == "landscapyml.__main__" and node.name in CLICK_COMMANDS:
                    name = f"{name}.callback"
                names.append(name)
                continue
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue

            class_name = f"{module}.{node.name}"
            names.append(class_name)
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if (
                    child.name.startswith("_")
                    and child.name not in PUBLIC_MAGIC_METHODS
                ):
                    continue
                names.append(f"{class_name}.{child.name}")
    return names


def test_pydocstyle_numpy_convention() -> None:
    """Require configured pydocstyle checks across the source package."""
    subprocess.run(
        [sys.executable, "-m", "pydocstyle", "src/landscapyml"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def test_numpydoc_public_api() -> None:
    """Reject structural NumPy-docstring errors on public API objects."""
    failures: list[str] = []
    for name in _public_object_names():
        allowed = IGNORED_NUMPYDOC_CODES | OBJECT_CODE_EXCLUSIONS.get(name, set())
        errors = [
            error for error in validate(name)["errors"] if error[0] not in allowed
        ]
        failures.extend(f"{name}: {code} {message}" for code, message in errors)

    assert not failures, "\n" + "\n".join(failures)
