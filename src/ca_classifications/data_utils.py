from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "This helper requires pandas. Install with `pip install pandas` or the project's tracking extra."
        ) from exc


@dataclass
class CSVConfigRequest:
    csv_path: Path
    sequence_column: str
    label_column: str
    embedding_mode: str = "hard"
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    max_epochs: int = 5
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    seed: Optional[int] = None
    val_split: float = 0.0
    val_seed: Optional[int] = None
    model_key: str = "sequence_gp_classifier"
    # TODO: Support landscapy FitnessLandscape inputs directly.


def build_config_from_dataframe(
    df: Any,
    wandb_project: Optional[str] = None,
    *,
    sequence_column: str,
    label_column: str,
    model: str = "sequence_gp_classifier",
    data: str = "raw_sequences",
    embedding_mode: str = "hard",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    max_epochs: int = 5,
    use_wandb: bool = True,
    wandb_run_name: Optional[str] = None,
    seed: Optional[int] = None,
    val_split: float = 0.0,
    val_seed: Optional[int] = None,
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct a config dictionary from an in-memory dataframe."""
    if sequence_column not in df or label_column not in df:
        raise ValueError(f"Columns '{sequence_column}' and '{label_column}' must exist in the dataframe.")

    seqs = df[sequence_column].tolist()
    cats = df[label_column].astype("category")
    labels = cats.cat.codes.tolist()
    label_mapping = cats.cat.categories.tolist()

    config: Dict[str, Any] = {
        "model": model_key or model,
        "data": data,
        "model_kwargs": {"num_classes": int(len(label_mapping))},
        "data_kwargs": {
            "train_sequences": seqs,
            "train_labels": labels,
            "label_key": label_column,
            "label_mapping": label_mapping,
            "embedding_mode": embedding_mode,
            "model_name": model_name,
            "val_split": val_split,
            "val_seed": val_seed,
        },
        "trainer_kwargs": {
            "max_epochs": max_epochs,
            "log_dir": "logs",
            "checkpoint_dir": "checkpoints",
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
        },
        "seed": seed,
        "fit": True,
        "test": False,
    }
    return config


def build_config_from_csv(req: CSVConfigRequest) -> Dict[str, Any]:
    """Load a CSV and build a training config."""
    _require_pandas()
    import pandas as pd  # type: ignore

    df = pd.read_csv(req.csv_path)
    return build_config_from_dataframe(
        df,
        sequence_column=req.sequence_column,
        label_column=req.label_column,
        embedding_mode=req.embedding_mode,
        model_name=req.model_name,
        max_epochs=req.max_epochs,
        use_wandb=req.use_wandb,
        wandb_project=req.wandb_project,
        wandb_run_name=req.wandb_run_name,
        seed=req.seed,
        val_split=req.val_split,
        val_seed=req.val_seed,
        model_key=req.model_key,
    )


def write_config(config: Mapping[str, Any], path: Path) -> None:
    """Write a config mapping to JSON (Hydra-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
