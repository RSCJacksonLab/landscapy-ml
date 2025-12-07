from pathlib import Path

import torch

from landscapyml.config import JobConfig
from landscapyml.hydra_runner import _config_path, _run_job


def test_run_job_executes_without_training(tmp_path):
    records = [
        {"sequence_tensor": torch.tensor([1.0, 0.0]), "fitness_tensors": {"label": 0}},
        {"sequence_tensor": torch.tensor([0.0, 1.0]), "fitness_tensors": {"label": 1}},
    ]
    cfg = JobConfig(
        model="sequence_gp_classifier",
        data="fitness_landscape_records",
        model_kwargs={"num_classes": 2, "num_inducing": 2, "num_data": 2},
        data_kwargs={
            "train_data": records,
            "label_key": "label",
            "label_mapping": ["a", "b"],
        },
        trainer_kwargs={
            "max_epochs": 1,
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "use_wandb": False,
        },
        fit=False,
        test=False,
    )
    _run_job(cfg, base_dir=Path(tmp_path))
    # Ensure label mapping was persisted next to checkpoints
    mapping = tmp_path / "ckpts" / "label_mapping.json"
    assert mapping.exists()


def test_config_path_defaults_to_packaged_conf(monkeypatch):
    monkeypatch.delenv("LANDSCAPYML_CONFIG_PATH", raising=False)
    conf_dir = Path(_config_path())
    assert conf_dir.joinpath("config.yaml").is_file()
    assert "landscapyml" in conf_dir.parts
