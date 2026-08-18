import pytest
import pytorch_lightning as pl
import torch

import landscapyml.core.trainer as trainer_module
from landscapyml.core.model_registry import register_model
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


def test_create_trainer_enables_available_wandb(
    monkeypatch, stub_trainer_dependencies
):
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
            raise ModuleNotFoundError("No module named 'wandb'")

    monkeypatch.setattr(pl.loggers, "WandbLogger", MissingWandbLogger)

    with pytest.warns(RuntimeWarning, match="tracking"):
        trainer = create_trainer(
            use_wandb=True,
            wandb_project="project",
            checkpoint_monitor=None,
        )

    assert len(trainer.kwargs["logger"]) == 1


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
