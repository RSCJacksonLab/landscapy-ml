import pytorch_lightning as pl
import torch

from landscapyml.core.model_registry import register_model
from landscapyml.core.trainer import TrainingJob


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
