import warnings
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch

import landscapyml.core.trainer as trainer_module
from landscapyml.core.model_registry import register_data, register_model
from landscapyml.core.trainer import TrainingJob, create_trainer


class StubTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubTensorBoardLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def stub_trainer_dependencies(monkeypatch):
    monkeypatch.setattr(trainer_module.pl, "Trainer", StubTrainer)
    monkeypatch.setattr(trainer_module, "TensorBoardLogger", StubTensorBoardLogger)
    monkeypatch.setattr(
        trainer_module.pl.callbacks,
        "ModelCheckpoint",
        lambda **kwargs: ("checkpoint", kwargs),
    )


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


def test_create_trainer_does_not_initialize_wandb_by_default(
    monkeypatch, stub_trainer_dependencies
):
    def fail_if_called(**kwargs):  # noqa: ARG001
        raise AssertionError("W&B must not be initialized by default")

    monkeypatch.setattr(pl.loggers, "WandbLogger", fail_if_called)

    trainer = create_trainer(checkpoint_monitor=None)

    assert len(trainer.kwargs["logger"]) == 1


def test_create_trainer_explicitly_disables_wandb(
    monkeypatch, stub_trainer_dependencies
):
    def fail_if_called(**kwargs):  # noqa: ARG001
        raise AssertionError("W&B must not be initialized when disabled")

    monkeypatch.setattr(pl.loggers, "WandbLogger", fail_if_called)

    trainer = create_trainer(use_wandb=False, checkpoint_monitor=None)

    assert len(trainer.kwargs["logger"]) == 1


def test_create_trainer_enables_available_wandb(monkeypatch, stub_trainer_dependencies):
    calls = []

    class StubWandbLogger:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(pl.loggers, "WandbLogger", StubWandbLogger)

    trainer = create_trainer(
        use_wandb=True,
        wandb_project="project",
        checkpoint_monitor=None,
    )

    assert calls == [
        {
            "project": "project",
            "entity": None,
            "name": None,
            "tags": None,
            "save_dir": None,
        }
    ]
    assert len(trainer.kwargs["logger"]) == 2


def test_create_trainer_warns_when_wandb_dependency_is_absent(
    monkeypatch, stub_trainer_dependencies
):
    class MissingWandbLogger:
        def __init__(self, **kwargs):  # noqa: ARG002
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

    monkeypatch.setattr(pl.loggers, "WandbLogger", MissingWandbLogger)

    with pytest.warns(RuntimeWarning, match="tracking"):
        trainer = create_trainer(
            use_wandb=True,
            wandb_project="project",
            checkpoint_monitor=None,
        )

    assert len(trainer.kwargs["logger"]) == 1


def test_create_trainer_propagates_nested_wandb_import_failures(
    monkeypatch, stub_trainer_dependencies
):
    class BrokenWandbLogger:
        def __init__(self, **kwargs):  # noqa: ARG002
            raise ModuleNotFoundError("No module named 'requests'", name="requests")

    monkeypatch.setattr(pl.loggers, "WandbLogger", BrokenWandbLogger)

    with pytest.raises(ModuleNotFoundError, match="requests"):
        create_trainer(
            use_wandb=True,
            wandb_project="project",
            checkpoint_monitor=None,
        )


def test_create_trainer_warns_for_missing_wandb_project(
    monkeypatch, stub_trainer_dependencies
):
    class StubWandbLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(pl.loggers, "WandbLogger", StubWandbLogger)

    with pytest.warns(RuntimeWarning, match="wandb_project is None"):
        trainer = create_trainer(use_wandb=True, checkpoint_monitor=None)

    assert len(trainer.kwargs["logger"]) == 2


def test_training_job_does_not_retry_genuine_setup_failures():
    class BrokenDataModule:
        def __init__(self):
            self.setup_calls = 0

        def setup(self, stage=None):  # noqa: ARG002
            self.setup_calls += 1
            raise RuntimeError("invalid training records")

    datamodule = BrokenDataModule()
    register_model("setup_error_model", lambda: None, overwrite=True)
    register_data("setup_error_data", lambda: datamodule, overwrite=True)
    job = TrainingJob(
        model_name="setup_error_model",
        data_name="setup_error_data",
    )

    with pytest.raises(RuntimeError, match="invalid training records"):
        job._build_datamodule()

    assert datamodule.setup_calls == 1


def test_training_job_supports_setup_without_stage_argument():
    class LegacyDataModule:
        def __init__(self):
            self.setup_calls = 0

        def setup(self):
            self.setup_calls += 1

    datamodule = LegacyDataModule()
    register_model("legacy_setup_model", lambda: None, overwrite=True)
    register_data("legacy_setup_data", lambda: datamodule, overwrite=True)
    job = TrainingJob(
        model_name="legacy_setup_model",
        data_name="legacy_setup_data",
    )

    assert job._build_datamodule() is datamodule
    assert datamodule.setup_calls == 1


def test_training_job_warns_when_metadata_logging_fails():
    class DummyDataModule:
        train_records = [{}]
        label_mapping = None
        label_key = "label"
        val_split = 0.0

        def setup(self, stage=None):  # noqa: ARG002
            pass

        def train_dataloader(self):
            return [torch.zeros(1, 2)]

    class TinyModel(pl.LightningModule):
        def __init__(self, num_features):
            super().__init__()
            self.num_features = num_features

    class BrokenLogger:
        def log_hyperparams(self, metadata):  # noqa: ARG002
            raise RuntimeError("logger unavailable")

    logger = BrokenLogger()
    register_model("metadata_warning_model", TinyModel, overwrite=True)
    register_data("metadata_warning_data", DummyDataModule, overwrite=True)
    job = TrainingJob(
        model_name="metadata_warning_model",
        data_name="metadata_warning_data",
        trainer_factory=lambda **kwargs: SimpleNamespace(  # noqa: ARG005
            logger=logger,
            loggers=[logger],
        ),
    )

    with pytest.warns(RuntimeWarning, match="Failed to log training metadata"):
        job.build()


def test_training_job_validates_registry_names():
    with pytest.raises(ValueError, match="Unknown model"):
        TrainingJob(model_name="missing-model", data_name="landscape_records")
    with pytest.raises(ValueError, match="Unknown data builder"):
        TrainingJob(model_name="external", data_name="missing-data")


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

    with warnings.catch_warnings():
        warnings.simplefilter("error")
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
