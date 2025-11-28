from __future__ import annotations

import os
from pathlib import Path
from typing import List

import click

from .hydra_runner import ENV_CONFIG_PATH, run_with_hydra
from .trainer import _DATA_REGISTRY, _MODEL_REGISTRY  # type: ignore
from .data_utils import build_config_from_csv, write_config, CSVConfigRequest


@click.group(help="landscapy-ml CLI (training utilities).")
def cli() -> None:
    pass


@cli.command("list", help="List registered models and data builders.")
def list_registered() -> None:
    click.echo("Models:")
    for name in sorted(_MODEL_REGISTRY):
        click.echo(f"  - {name}")
    click.echo("\nData builders:")
    for name in sorted(_DATA_REGISTRY):
        click.echo(f"  - {name}")


@cli.command(
    "train",
    help="Run training via Hydra-configured TrainingJob. "
    "Additional Hydra overrides can be passed after a '--', e.g., '-- +trainer_kwargs.max_epochs=5'.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Config directory or path to config.yaml for Hydra (defaults to project conf/).",
)
@click.pass_context
def train(ctx: click.Context, config_path: Path | None) -> None:
    # Forward any remaining args directly to Hydra as overrides.
    overrides = list(ctx.args)
    if config_path:
        cfg_path = config_path.resolve()
        if cfg_path.is_file():
            if cfg_path.name != "config.yaml":
                raise click.ClickException(
                    "When providing a file path, it must be named 'config.yaml' for Hydra."
                )
            cfg_path = cfg_path.parent
        os.environ[ENV_CONFIG_PATH] = str(cfg_path)
    code = run_with_hydra(overrides=overrides)
    if code:
        raise SystemExit(code)


@cli.command(
    "config-from-csv",
    help="Construct a Hydra config from a CSV with sequences and labels.",
)
@click.option("--csv-path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sequence-column", required=True, help="Column name containing raw sequences.")
@click.option("--label-column", required=True, help="Column name containing class labels.")
@click.option("--out-dir", type=click.Path(dir_okay=True, file_okay=False, path_type=Path), default=Path("landscapyml_run_conf"), show_default=True)
@click.option("--embedding-mode", default="hard", show_default=True, help="Embedding mode: hard or soft.")
@click.option("--model-name", default="facebook/esm2_t6_8M_UR50D", show_default=True, help="PLM model for embeddings.")
@click.option("--max-epochs", default=5, show_default=True, help="Max training epochs.")
@click.option("--use-wandb/--no-wandb", default=True, show_default=True, help="Enable Weights & Biases logging.")
@click.option("--wandb-project", default=None, help="W&B project name (None = use wandb default).")
@click.option("--wandb-run-name", default=None, help="W&B run name.")
@click.option("--seed", default=None, type=int, help="Optional seed.")
@click.option("--val-split", default=0.1, show_default=True, type=float, help="Fraction of training used for validation when no val data provided.")
@click.option("--val-seed", default=None, type=int, help="Optional seed for val split.")
@click.option("--model-key", default="sequence_gp_classifier", show_default=True, help="Registered model key to use (e.g., sequence_gp_classifier, sequence_mlp_classifier).")
def config_from_csv(
    csv_path: Path,
    sequence_column: str,
    label_column: str,
    out_dir: Path,
    embedding_mode: str,
    model_name: str,
    max_epochs: int,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_run_name: str | None,
    seed: int | None,
    val_split: float,
    val_seed: int | None,
    model_key: str,
) -> None:
    req = CSVConfigRequest(
        csv_path=csv_path,
        sequence_column=sequence_column,
        label_column=label_column,
        embedding_mode=embedding_mode,
        model_name=model_name,
        max_epochs=max_epochs,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        seed=seed,
        val_split=val_split,
        val_seed=val_seed,
        model_key=model_key,
    )
    config = build_config_from_csv(req)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, out_dir / "config.yaml")
    click.echo(f"Wrote config to {out_dir / 'config.yaml'}")


def main(argv: List[str] | None = None) -> int:
    try:
        cli.main(args=argv, prog_name="landscapyml", standalone_mode=False)
    except SystemExit as exc:  # click exits via SystemExit
        return exc.code
    except Exception:
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
