from __future__ import annotations


def is_missing_optional_dependency(
    error: ModuleNotFoundError,
    package: str,
) -> bool:
    """Return whether an import failed because the optional root is absent."""

    return error.name == package
