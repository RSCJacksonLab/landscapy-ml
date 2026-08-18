from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

import landscapyml.core.trainer as trainer_module
from landscapyml.core.model_registry import register_data, register_model
from landscapyml.core.trainer import TrainingJob, create_trainer


def test_training_job_builds_components(tmp_path):
    class TinyModel(pl.LightningModule):
        def __init__(self, num_features: int):
            super().__init__()
            self.linear = torch.nn.Linear(num_features, 1)

    records = [
        {"sequence_tensor": torch.tensor([1.0, 0.0]), "fitness_tensors": {"label": 0}},
        {"sequence_tensor": torch.tensor([0.0, 1.0]), "fitness_tensors": {"label": 1}},
    ]
    ckpt_dir = tmp_path / "ckpts"
    log_dir = tmp_path / "logs"
    register_model("tiny_test_model", TinyModel, overwrite=True)
    job = TrainingJob(
        model_name="tiny_test_model",
        data_name="landscape_records",
        data_kwargs={"train_data": records},
        trainer_kwargs={
            "max_epochs": 1,
            "checkpoint_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
            "use_wandb": False,
        },
        seed=123,
    )
    trainer, model, dm = job.run(fit=False, test=False)
    assert trainer.max_epochs == 1
    assert len(dm.train_records) == 2
    assert isinstance(model, TinyModel)
    assert model.linear.in_features == 2


def test_create_trainer_builds_tensorboard_and_optional_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    class FakeTensorBoardLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_trainer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(trainer_module, "TensorBoardLogger", FakeTensorBoardLogger)
    monkeypatch.setattr(trainer_module.pl.callbacks, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(trainer_module.pl, "Trainer", fake_trainer)

    trainer = create_trainer(
        max_epochs=3,
        use_wandb=False,
        checkpoint_monitor="val/mae",
        checkpoint_dir="ckpts",
    )

    assert trainer.max_epochs == 3
    assert len(captured["logger"]) == 1
    assert captured["callbacks"][0].kwargs["monitor"] == "val/mae"

    create_trainer(use_wandb=False, checkpoint_monitor=None)
    assert captured["callbacks"] == []


def test_create_trainer_builds_explicit_wandb_logger(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(trainer_module, "TensorBoardLogger", FakeLogger)
    monkeypatch.setattr(trainer_module.pl.loggers, "WandbLogger", FakeLogger)
    monkeypatch.setattr(
        trainer_module.pl,
        "Trainer",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    with pytest.warns(RuntimeWarning, match="wandb_project is None"):
        trainer = create_trainer(use_wandb=True, checkpoint_monitor=None)

    assert len(trainer.logger) == 2


def test_training_job_validates_registry_names():
    with pytest.raises(ValueError, match="Unknown model"):
        TrainingJob(model_name="missing-model", data_name="landscape_records")
    with pytest.raises(ValueError, match="Unknown data builder"):
        TrainingJob(model_name="external", data_name="missing-data")


def test_training_job_normalizes_splits_and_rejects_conflicts():
    captured = {}

    class DataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            self.stage = stage

    class Model(pl.LightningModule):
        pass

    def data_factory(*, train_indices=None):
        captured["train_indices"] = train_indices
        return DataModule()

    register_data("split-aware-test", data_factory, overwrite=True)
    register_model(
        "split-model-test",
        Model,
        overwrite=True,
        requires_num_features=False,
    )
    trainer = SimpleNamespace(logger=None, loggers=[])
    job = TrainingJob(
        model_name="split-model-test",
        data_name="split-aware-test",
        split_indices={"training": [0, 1]},
        trainer_factory=lambda **kwargs: trainer,
    )

    _, _, dm = job.build()

    assert captured["train_indices"] == [0, 1]
    assert dm.stage == "fit"

    conflicting = TrainingJob(
        model_name="split-model-test",
        data_name="split-aware-test",
        data_kwargs={"train_indices": [0]},
        split_indices={"train": [1]},
        trainer_factory=lambda **kwargs: trainer,
    )
    with pytest.raises(ValueError, match="both in data_kwargs"):
        conflicting.build()


def test_training_job_rejects_splits_for_non_split_aware_factory():
    class Model(pl.LightningModule):
        pass

    register_data(
        "no-splits-test",
        lambda: SimpleNamespace(setup=lambda stage=None: None),
        overwrite=True,
    )
    register_model(
        "no-splits-model-test",
        Model,
        overwrite=True,
        requires_num_features=False,
    )
    job = TrainingJob(
        model_name="no-splits-model-test",
        data_name="no-splits-test",
        split_indices={"test": [0]},
    )

    with pytest.raises(ValueError, match="does not accept pre-defined"):
        job.build()
