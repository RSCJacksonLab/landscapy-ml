from landscapyml.core._optional import is_missing_optional_dependency


def test_optional_dependency_detection_rejects_nested_missing_import():
    root_error = ModuleNotFoundError(
        "No module named 'fitness_landscape'",
        name="fitness_landscape",
    )
    nested_error = ModuleNotFoundError(
        "No module named 'sklearn'",
        name="sklearn",
    )

    assert is_missing_optional_dependency(root_error, "fitness_landscape")
    assert not is_missing_optional_dependency(nested_error, "fitness_landscape")
