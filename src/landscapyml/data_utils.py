from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _require_pandas():
    """Ensure pandas is available, raising an informative ImportError otherwise."""
    try:
        import pandas as pd  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "This helper requires pandas. Install with `pip install pandas` or the project's tracking extra."
        ) from exc


@dataclass
class CSVConfigRequest:
    """
    Structured request for building a training config from a CSV file.

    Attributes
    ----------
    csv_path : Path
        Path to the input CSV file.
    sequence_column : str
        Name of the column containing raw sequences.
    label_column : str
        Name of the column containing labels.
    embedding_mode : str, default="hard"
        Embedding strategy to use for raw sequences.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        Model identifier used by landscapy embedders.
    max_epochs : int, default=5
        Maximum number of training epochs.
    use_wandb : bool, default=True
        Whether to enable Weights & Biases logging.
    wandb_project : str, optional
        Weights & Biases project name.
    wandb_run_name : str, optional
        Optional run name for Weights & Biases.
    seed : int, optional
        Optional global seed.
    val_split : float, default=0.0
        Fraction of training data held out for validation when validation data is not provided.
    val_seed : int, optional
        Seed controlling the validation split.
    model_key : str, default="sequence_gp_classifier"
        Registry key for the model factory to use.
    """

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
    """
    Construct a Hydra-compatible config from an in-memory dataframe.

    Parameters
    ----------
    df : Any
        Dataframe-like object containing sequence and label columns.
    wandb_project : str, optional
        Optional Weights & Biases project name.
    sequence_column : str
        Column name containing raw sequences.
    label_column : str
        Column name containing labels.
    model : str, default="sequence_gp_classifier"
        Registry key for the model factory.
    data : str, default="raw_sequences"
        Registry key for the data builder.
    embedding_mode : str, default="hard"
        Embedding strategy for raw sequences.
    model_name : str, default="facebook/esm2_t6_8M_UR50D"
        Model identifier used by landscapy embedders.
    max_epochs : int, default=5
        Maximum number of training epochs.
    use_wandb : bool, default=True
        Whether to enable Weights & Biases logging.
    wandb_run_name : str, optional
        Optional Weights & Biases run name.
    seed : int, optional
        Optional global seed.
    val_split : float, default=0.0
        Fraction of training data used for validation when validation data is not supplied.
    val_seed : int, optional
        Seed controlling the validation split.
    model_key : str, optional
        Alternate registry key for the model factory; overrides ``model`` if provided.

    Returns
    -------
    dict[str, Any]
        Hydra-ready configuration mapping for ``TrainingJob``.

    Raises
    ------
    ValueError
        If the expected sequence or label columns are missing from the dataframe.
    """
    if sequence_column not in df or label_column not in df:
        raise ValueError(
            f"Columns '{sequence_column}' and '{label_column}' must exist in the dataframe."
        )

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
    """
    Load a CSV and build a Hydra-compatible training config.

    Parameters
    ----------
    req : CSVConfigRequest
        Structured request describing CSV location and config options.

    Returns
    -------
    dict[str, Any]
        Hydra-ready configuration mapping.
    """
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
    """
    Write a configuration mapping to JSON for Hydra consumption.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration to serialize.
    path : Path
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
