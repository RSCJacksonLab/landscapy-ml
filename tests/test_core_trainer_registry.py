import sys
import types

import pytest
import pytorch_lightning as pl

from landscapyml.core.trainer import (
    build_external_model,
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
