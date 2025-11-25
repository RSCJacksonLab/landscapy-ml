from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import json
from pathlib import Path
import pytorch_lightning as pl
import torch

from .data import SequenceClassificationDataModule, embed_sequences_to_records
from .gp_classification import SequenceGPClassifier, create_trainer
from .mlp_classification import SequenceMLPClassifier, SequenceMLPEnsembleClassifier

ModelFactory = Callable[..., pl.LightningModule]
DataFactory = Callable[..., pl.LightningDataModule]
TrainerFactory = Callable[..., pl.Trainer]

_MODEL_REGISTRY: Dict[str, ModelFactory] = {}
_DATA_REGISTRY: Dict[str, DataFactory] = {}


def register_model(name: str, factory: ModelFactory, *, overwrite: bool = False) -> None:
    if name in _MODEL_REGISTRY and not overwrite:
        raise ValueError(f"Model '{name}' is already registered.")
    _MODEL_REGISTRY[name] = factory


def register_data(name: str, factory: DataFactory, *, overwrite: bool = False) -> None:
    if name in _DATA_REGISTRY and not overwrite:
        raise ValueError(f"Data builder '{name}' is already registered.")
    _DATA_REGISTRY[name] = factory


# Default registry entries
register_model("sequence_gp_classifier", SequenceGPClassifier, overwrite=True)
register_model("sequence_mlp_classifier", SequenceMLPClassifier, overwrite=True)
register_model("sequence_mlp_ensemble", SequenceMLPEnsembleClassifier, overwrite=True)
# TODO: add additional models to the registry and expose selection in CLI/config.
register_data("fitness_landscape_records", SequenceClassificationDataModule, overwrite=True)
register_data(
    "raw_sequences",
    SequenceClassificationDataModule.from_sequences,
    overwrite=True,
)
register_data(
    "fitness_landscape",
    SequenceClassificationDataModule,
    overwrite=True,
)
# TODO: Register landscapy-native builders when available.


@dataclass
class TrainingJob:
    """
    Helper to construct and run training jobs in a consistent way.

    This class keeps factories for models/data modules, builds them from
    kwargs, and wires up a Lightning Trainer. A CLI can map parsed args to
    the *_kwargs fields and pick a registry key for model/data.
    """

    model_name: str
    data_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    trainer_factory: TrainerFactory = create_trainer

    def __post_init__(self) -> None:
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
        dm = factory(**self.data_kwargs)
        # Ensure datasets are built before we inspect shapes or hand to Trainer.
        try:
            dm.setup("fit")
        except Exception:
            dm.setup()
        return dm

    def _infer_num_features(self, dm: pl.LightningDataModule) -> int:
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        features = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not torch.is_tensor(features):
            raise RuntimeError("Expected tensor features to infer num_features.")
        return int(features.shape[-1])

    def _build_model(self, dm: pl.LightningDataModule) -> pl.LightningModule:
        factory = _MODEL_REGISTRY[self.model_name]
        kwargs = dict(self.model_kwargs)
        if "num_features" not in kwargs:
            kwargs["num_features"] = self._infer_num_features(dm)
        try:
            return factory(**kwargs)
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
        for logger in trainer.loggers if isinstance(trainer.logger, list) else [trainer.logger]:
            try:
                logger.log_hyperparams(metadata)
            except Exception:
                continue

        # Persist label mapping next to checkpoints if available
        if label_mapping is not None:
            ckpt_dir = self.trainer_kwargs.get("checkpoint_dir") or "checkpoints"
            try:
                path = Path(ckpt_dir)
                path.mkdir(parents=True, exist_ok=True)
                with path.joinpath("label_mapping.json").open("w") as fh:
                    json.dump(list(label_mapping), fh)
            except Exception:
                pass
        return trainer, model, dm

    def run(
        self,
        *,
        fit: bool = True,
        test: bool = False,
    ) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
        trainer, model, dm = self.build()
        if fit:
            trainer.fit(model, datamodule=dm)
        if test:
            trainer.test(model, datamodule=dm)
        return trainer, model, dm
