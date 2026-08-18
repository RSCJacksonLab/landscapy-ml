from __future__ import annotations


def is_missing_optional_dependency(
    error: ModuleNotFoundError,
    package: str,
) -> bool:
    """Test whether an import failed at an optional package root.

    Parameters
    ----------
    error : ModuleNotFoundError
        Import exception to inspect.
    package : str
        Optional top-level package name.

    Returns
    -------
    bool
        ``True`` only when ``error.name`` exactly matches ``package``.
    """
    return error.name == package
