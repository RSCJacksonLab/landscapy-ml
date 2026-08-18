"""Construct Lightning trainers and registry-driven training jobs."""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from ._optional import is_missing_optional_dependency
from .model_registry import (
    _DATA_REGISTRY,
    _MODEL_REGISTRY,
    factory_accepts_kwargs,
    normalize_split_indices,
    register_builtin_components,
)

TrainerFactory = Callable[..., pl.Trainer]


def create_trainer(
    *,
    max_epochs: int = 50,
    accelerator: Optional[str] = "auto",
    devices: Optional[int] = 1,
    log_every_n_steps: int = 10,
    log_dir: str = "logs",
    experiment_name: str = "landscapyml",
    checkpoint_dir: str = "checkpoints",
    checkpoint_monitor: Optional[str] = "val/loss",
    checkpoint_mode: str = "min",
    checkpoint_every_n_epochs: int = 1,
    save_top_k: int = 1,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None,
    wandb_dir: Optional[str] = None,
    num_sanity_val_steps: int = 0,
) -> pl.Trainer:
    """
    Build a PyTorch Lightning ``Trainer`` with TensorBoard (and optional W&B) logging.

    Parameters
    ----------
    max_epochs : int, default=50
        Maximum number of training epochs.
    accelerator : str or None, default="auto"
        Accelerator passed to Lightning (e.g., ``"cpu"``, ``"gpu"``, or ``"auto"``).
    devices : int or None, default=1
        Number of devices to use; forwarded to Lightning.
    log_every_n_steps : int, default=10
        Logging frequency in steps.
    log_dir : str, default="logs"
        Base directory for TensorBoard logs.
    experiment_name : str, default="landscapyml"
        Experiment subdirectory name used by loggers.
    checkpoint_dir : str, default="checkpoints"
        Directory for model checkpoints.
    checkpoint_monitor : str or None, default="val/loss"
        Metric name to monitor for checkpointing. ``None`` disables checkpointing.
    checkpoint_mode : str, default="min"
        Whether to minimize or maximize the monitored metric.
    checkpoint_every_n_epochs : int, default=1
        Frequency in epochs for checkpointing.
    save_top_k : int, default=1
        Number of best checkpoints to keep.
    use_wandb : bool, default=False
        Whether to opt in to Weights & Biases logging. The ``tracking`` extra
        must be installed when this is enabled.
    wandb_project : str, optional
        Optional W&B project name.
    wandb_entity : str, optional
        Optional W&B entity/organization.
    wandb_run_name : str, optional
        Optional W&B run name.
    wandb_tags : list[str], optional
        Optional W&B tags.
    wandb_dir : str, optional
        Optional W&B log directory.
    num_sanity_val_steps : int, default=0
        Number of validation sanity steps to run before training.

    Returns
    -------
    pytorch_lightning.Trainer
        Configured Lightning trainer instance.
    """
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False,
    )
    loggers = [tensorboard_logger]

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            if not is_missing_optional_dependency(exc, "wandb"):
                raise
            warnings.warn(
                f"W&B logging requested but wandb is not available: {exc}. "
                "Proceeding with TensorBoard only.",
                RuntimeWarning,
            )
        else:
            if wandb_project is None:
                warnings.warn(
                    "wandb_project is None; WandbLogger will use the default W&B project.",
                    RuntimeWarning,
                )
            try:
                wandb_logger = WandbLogger(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    tags=wandb_tags,
                    save_dir=wandb_dir,
                )
            except ModuleNotFoundError as exc:
                if not is_missing_optional_dependency(exc, "wandb"):
                    raise
                warnings.warn(
                    f"W&B logging requested but wandb is not available: {exc}. "
                    "Install landscapy-ml[tracking] to enable it. Proceeding "
                    "with TensorBoard only.",
                    RuntimeWarning,
                )
            else:
                loggers.append(wandb_logger)

    callbacks = []
    if checkpoint_monitor is not None:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                every_n_epochs=checkpoint_every_n_epochs,
                save_top_k=save_top_k,
            )
        )

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        logger=loggers,
        callbacks=callbacks,
        num_sanity_val_steps=num_sanity_val_steps,
    )


@dataclass
class TrainingJob:
    """
    Helper to construct and run training jobs in a consistent way.

    The class wires together model/data/trainer factories and optional seeding,
    mirroring how the CLI maps user-provided keyword arguments to registry keys.

    Parameters
    ----------
    model_name : str
        Registry key for the model factory.
    data_name : str
        Registry key for the data builder.
    model_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the model factory.
    data_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the data builder.
    split_indices : mapping[str, sequence[int]], optional
        Optional pre-defined train/validation/test indices. Missing splits are
        allowed and are delegated to the data builder fallback behavior.
    trainer_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the trainer factory.
    seed : int, optional
        Optional global seed for deterministic runs.
    trainer_factory : TrainerFactory, default=create_trainer
        Factory used to build the Lightning trainer.
    """

    model_name: str
    data_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_kwargs: Dict[str, Any] = field(default_factory=dict)
    split_indices: Optional[Mapping[str, Sequence[int]]] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    trainer_factory: TrainerFactory = create_trainer

    def __post_init__(self) -> None:
        """Validate registry keys after dataclass initialization.

        Raises
        ------
        ValueError
            If the requested model or data-builder name is unregistered.

        Notes
        -----
        Built-in registry entries are initialized before validation.
        """
        register_builtin_components()
        if self.model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Available: {', '.join(sorted(_MODEL_REGISTRY)) or 'none'}"
            )
        if self.data_name not in _DATA_REGISTRY:
            raise ValueError(
                f"Unknown data builder '{self.data_name}'. "
                f"Available: {', '.join(sorted(_DATA_REGISTRY)) or 'none'}"
            )

    def _build_datamodule(self) -> pl.LightningDataModule:
        factory = _DATA_REGISTRY[self.data_name]
        data_kwargs = dict(self.data_kwargs)
        normalized_splits = normalize_split_indices(self.split_indices)
        if normalized_splits:
            if not factory_accepts_kwargs(factory, tuple(normalized_splits)):
                accepted = ", ".join(sorted(normalized_splits))
                raise ValueError(
                    f"Data builder '{self.data_name}' does not accept pre-defined "
                    f"split indices ({accepted}). Choose a split-aware data builder "
                    "or omit split_indices."
                )
            overlap = sorted(set(data_kwargs).intersection(normalized_splits))
            if overlap:
                joined = ", ".join(overlap)
                raise ValueError(
                    f"Split indices were supplied both in data_kwargs and split_indices: {joined}."
                )
            data_kwargs.update(normalized_splits)
        dm = factory(**data_kwargs)
        # Ensure datasets are built before we inspect shapes or hand to Trainer.
        try:
            inspect.signature(dm.setup).bind("fit")
        except (TypeError, ValueError):
            dm.setup()
        else:
            dm.setup("fit")
        return dm

    def _infer_num_features(self, dm: pl.LightningDataModule) -> int:
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        features = batch[0] if isinstance(batch, (tuple, list)) else batch
        if torch.is_tensor(features):
            return int(features.shape[-1])
        if hasattr(features, "x") and torch.is_tensor(features.x):
            return int(features.x.shape[-1])
        raise RuntimeError("Expected tensor features or a graph batch with '.x'.")

    def _build_model(self, dm: pl.LightningDataModule) -> pl.LightningModule:
        entry = _MODEL_REGISTRY[self.model_name]
        kwargs = dict(self.model_kwargs)
        if entry.requires_num_features and "num_features" not in kwargs:
            kwargs["num_features"] = self._infer_num_features(dm)
        try:
            return entry.factory(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Failed to build model '{self.model_name}' with kwargs {kwargs!r}."
            ) from exc

    def _build_trainer(self) -> pl.Trainer:
        # Fallback to pl.Trainer if caller passes trainer_factory=None
        if self.trainer_factory is None:
            return pl.Trainer(**self.trainer_kwargs)
        return self.trainer_factory(**self.trainer_kwargs)

    def build(self) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
        """Construct the trainer, model, and initialized data module.

        Returns
        -------
        trainer : pytorch_lightning.Trainer
            Configured trainer.
        model : pytorch_lightning.LightningModule
            Model built from the registered factory.
        datamodule : pytorch_lightning.LightningDataModule
            Fit-stage data module built from the registered factory.

        Raises
        ------
        RuntimeError
            If required model feature width cannot be inferred.
        TypeError
            If model construction rejects the resolved keyword arguments.
        ValueError
            If split indices are unsupported, duplicated, or invalid.

        Notes
        -----
        When ``seed`` is set, this method seeds Lightning before construction.
        It also attempts to log run metadata and persist categorical label
        mappings; failures in those auxiliary writes emit warnings rather than
        aborting construction.
        """
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
        dm = self._build_datamodule()
        model = self._build_model(dm)
        trainer = self._build_trainer()
        # Log basic run metadata to all loggers
        metadata = {
            "num_train": len(getattr(dm, "train_records", [])),
            "label_key": getattr(dm, "label_key", None),
            "model_name": self.model_name,
            "data_name": self.data_name,
            "val_split": getattr(dm, "val_split", 0.0),
        }
        label_mapping = getattr(dm, "label_mapping", None)
        if label_mapping is not None:
            metadata["label_mapping"] = list(label_mapping)
        for logger in (
            trainer.loggers if isinstance(trainer.logger, list) else [trainer.logger]
        ):
            if logger is None:
                continue
            try:
                logger.log_hyperparams(metadata)
            except Exception as exc:
                warnings.warn(
                    f"Failed to log training metadata with "
                    f"{type(logger).__name__}: {exc}",
                    RuntimeWarning,
                )

        # Persist label mapping next to checkpoints if available
        if label_mapping is not None:
            ckpt_dir = self.trainer_kwargs.get("checkpoint_dir") or "checkpoints"
            try:
                path = Path(ckpt_dir)
                path.mkdir(parents=True, exist_ok=True)
                with path.joinpath("label_mapping.json").open("w") as fh:
                    json.dump(list(label_mapping), fh)
            except Exception as exc:
                warnings.warn(
                    f"Failed to persist label mapping in {ckpt_dir!r}: {exc}",
                    RuntimeWarning,
                )
        return trainer, model, dm

    def run(
        self,
        *,
        fit: bool = True,
        test: bool = False,
    ) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
        """Build a training job and optionally execute fit and test loops.

        Parameters
        ----------
        fit : bool, default=True
            Call ``trainer.fit`` after construction.
        test : bool, default=False
            Call ``trainer.test`` after optional fitting.

        Returns
        -------
        trainer : pytorch_lightning.Trainer
            Configured trainer after requested execution.
        model : pytorch_lightning.LightningModule
            Constructed model.
        datamodule : pytorch_lightning.LightningDataModule
            Initialized data module.
        """
        trainer, model, dm = self.build()
        if fit:
            trainer.fit(model, datamodule=dm)
        if test:
            trainer.test(model, datamodule=dm)
        return trainer, model, dm


__all__ = ["TrainingJob", "create_trainer"]
