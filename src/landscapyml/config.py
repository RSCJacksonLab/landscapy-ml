from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobConfig:
    """
    Hydra-structured configuration for training jobs.

    Attributes
    ----------
    model : str
        Registry key for the model factory to build.
    data : str
        Registry key for the data builder.
    model_kwargs : Dict[str, Any]
        Keyword arguments passed to the model factory.
    data_kwargs : Dict[str, Any]
        Keyword arguments passed to the data builder.
    trainer_kwargs : Dict[str, Any]
        Keyword arguments passed to the trainer factory (e.g., `create_trainer`).
    log_file : Optional[str]
        Optional path to a log file for package-level logging.
    log_level : str
        Logging level to configure for the package logger.
    seed : Optional[int]
        Optional global seed for deterministic runs.
    fit : bool
        Whether to run training via ``Trainer.fit``.
    test : bool
        Whether to run evaluation via ``Trainer.test``.
    """

    model: str = "sequence_mlp_classifier"
    data: str = "raw_sequences"
    # TODO: extend model/data selections when more registry entries are added.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_kwargs: Dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)
    log_file: Optional[str] = None
    log_level: str = "INFO"
    seed: Optional[int] = None
    fit: bool = True
    test: bool = False
