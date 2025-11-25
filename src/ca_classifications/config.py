from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobConfig:
    """Hydra-structured config for training jobs."""

    model: str = "sequence_gp_classifier"
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
