import json
from pathlib import Path

import torch

from landscapyml.trainer import TrainingJob


def test_training_job_builds_components(tmp_path):
    records = [
        {"sequence_tensor": torch.tensor([1.0, 0.0]), "fitness_tensors": {"label": 0}},
        {"sequence_tensor": torch.tensor([0.0, 1.0]), "fitness_tensors": {"label": 1}},
    ]
    ckpt_dir = tmp_path / "ckpts"
    log_dir = tmp_path / "logs"
    job = TrainingJob(
        model_name="sequence_gp_classifier",
        data_name="fitness_landscape_records",
        model_kwargs={"num_classes": 2, "num_inducing": 2, "num_data": 2},
        data_kwargs={
            "train_data": records,
            "label_key": "label",
            "label_mapping": ["a", "b"],
        },
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
    mapping_path = ckpt_dir / "label_mapping.json"
    assert mapping_path.exists()
    assert json.loads(mapping_path.read_text()) == ["a", "b"]
