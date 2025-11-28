from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import JobConfig
from .logging_utils import configure_logger
from .trainer import TrainingJob

# Register structured config so Hydra can validate keys.
cs = ConfigStore.instance()
cs.store(name="landscapyml_job", node=JobConfig)

# Resolve default config directory (project_root/conf).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf"
ENV_CONFIG_PATH = "LANDSCAPYML_CONFIG_PATH"


def _config_path() -> str:
    env_path = os.environ.get(ENV_CONFIG_PATH)
    if env_path:
        return env_path
    return str(DEFAULT_CONFIG_PATH)


def _make_absolute(path_str: Optional[str], base: Path) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(base / p)


def _run_job(cfg: JobConfig, base_dir: Optional[Path] = None) -> None:
    configure_logger(cfg.log_file, cfg.log_level)
    # TODO: honor model choices beyond the GP classifier when registry is expanded.
    if base_dir is None:
        base_dir = Path.cwd()
    # Normalize trainer paths relative to config dir
    tk = dict(cfg.trainer_kwargs)
    tk["log_dir"] = _make_absolute(tk.get("log_dir"), base_dir)
    tk["checkpoint_dir"] = _make_absolute(tk.get("checkpoint_dir"), base_dir)

    job = TrainingJob(
        model_name=cfg.model,
        data_name=cfg.data,
        model_kwargs=cfg.model_kwargs,
        data_kwargs=cfg.data_kwargs,
        trainer_kwargs=tk,
        seed=cfg.seed,
    )
    trainer, model, dm = job.build()
    if cfg.fit:
        trainer.fit(model, datamodule=dm)
    if cfg.test:
        trainer.test(model, datamodule=dm)


def run_with_hydra(overrides: Optional[Iterable[str]] = None) -> int:
    """
    Programmatic entry point to run the Hydra-configured training job.

    Overrides are standard Hydra key=value strings (e.g., 'model_kwargs.num_classes=4').
    """
    config_dir = Path(_config_path()).resolve()
    if not config_dir.is_dir():
        raise RuntimeError(f"Config path {config_dir} is not a directory containing config.yaml")

    over = list(overrides) if overrides else []

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="landscapyml"):
            cfg = compose(config_name="config", overrides=over)
        job_cfg = OmegaConf.to_object(cfg)
        if not isinstance(job_cfg, JobConfig):
            job_cfg = JobConfig(**job_cfg)  # type: ignore[arg-type]
        _run_job(job_cfg, base_dir=config_dir)
    except SystemExit as exc:
        return exc.code
    return 0


__all__ = ["run_with_hydra", "ENV_CONFIG_PATH"]
