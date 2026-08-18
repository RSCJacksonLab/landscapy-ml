import sys
import types

import pytest
import pytorch_lightning as pl

from landscapyml.core.model_registry import (
    available_data_builders,
    available_models,
    build_external_model,
    factory_accepts_kwargs,
    get_data_factory,
    get_model_entry,
    normalize_split_indices,
    register_data,
    register_model,
)


def test_register_model_duplicate_raises():
    def factory():
        return None

    name = "temp_model_entry"
    register_model(name, factory, overwrite=True)
    with pytest.raises(ValueError):
        register_model(name, factory, overwrite=False)


def test_register_data_duplicate_raises():
    def factory():
        return None

    name = "temp_data_entry"
    register_data(name, factory, overwrite=True)
    with pytest.raises(ValueError):
        register_data(name, factory, overwrite=False)


def test_build_external_model_with_class_path():
    module_name = "tmp_ext_module_for_test"
    module = types.ModuleType(module_name)

    class ExternalModel(pl.LightningModule):
        def __init__(self):
            super().__init__()

    module.ExternalModel = ExternalModel
    sys.modules[module_name] = module

    model = build_external_model(class_path=f"{module_name}.ExternalModel")
    assert isinstance(model, ExternalModel)


def test_registry_getters_and_sorted_available_names():
    def model_factory():
        return None

    def data_factory():
        return None

    register_model(
        "zz_test_model",
        model_factory,
        overwrite=True,
        requires_num_features=False,
    )
    register_data("zz_test_data", data_factory, overwrite=True)

    assert available_models() == sorted(available_models())
    assert available_data_builders() == sorted(available_data_builders())
    assert get_model_entry("zz_test_model").factory is model_factory
    assert get_model_entry("zz_test_model").requires_num_features is False
    assert get_data_factory("zz_test_data") is data_factory


def test_normalize_split_indices_aliases_and_validation():
    assert normalize_split_indices(None) == {}
    assert normalize_split_indices(
        {"TRAINING": ["1", 2], "validation": [3], "test": None}
    ) == {"train_indices": [1, 2], "val_indices": [3]}

    with pytest.raises(ValueError, match="Unknown split name"):
        normalize_split_indices({"holdout": [1]})
    with pytest.raises(ValueError, match="supplied more than once"):
        normalize_split_indices({"val": [1], "validation": [2]})


def test_factory_accepts_explicit_and_variadic_kwargs():
    def explicit(*, train_indices=None):
        return train_indices

    def variadic(**kwargs):
        return kwargs

    assert factory_accepts_kwargs(explicit, ["train_indices"]) is True
    assert factory_accepts_kwargs(explicit, ["test_indices"]) is False
    assert factory_accepts_kwargs(variadic, ["anything"]) is True
    assert factory_accepts_kwargs(object(), ["anything"]) is False


def test_build_external_model_validates_paths_adapters_and_result_type():
    module_name = "tmp_ext_module_validation"
    module = types.ModuleType(module_name)

    class PlainModel:
        def __init__(self, value=0):
            self.value = value

    class WrappedModel(pl.LightningModule):
        def __init__(self, model, offset=0):
            super().__init__()
            self.value = model.value + offset

    module.PlainModel = PlainModel
    module.WrappedModel = WrappedModel
    sys.modules[module_name] = module

    wrapped = build_external_model(
        class_path=f"{module_name}.PlainModel",
        init_kwargs={"value": 2},
        adapter_path=f"{module_name}.WrappedModel",
        adapter_kwargs={"offset": 3},
    )
    assert isinstance(wrapped, WrappedModel)
    assert wrapped.value == 5

    with pytest.raises(ValueError, match="fully qualified"):
        build_external_model(class_path="PlainModel")
    with pytest.raises(ImportError, match="Could not find"):
        build_external_model(class_path=f"{module_name}.Missing")
    with pytest.raises(TypeError, match="must be a LightningModule"):
        build_external_model(class_path=f"{module_name}.PlainModel")
