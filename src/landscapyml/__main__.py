from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import click

from .hydra_runner import ENV_CONFIG_PATH, run_with_hydra
from .core.trainer import _DATA_REGISTRY, _MODEL_REGISTRY  # type: ignore
from .data_utils import build_config_from_csv, write_config, CSVConfigRequest
from .landscape_regression import (
    LandscapeRegressionConfig,
    available_landscape_regression_runners,
    import_builtin_examples,
    run_landscape_regression,
)


def _extract_config_path_from_overrides(
    args: list[str],
) -> tuple[list[str], Path | None]:
    """
    If the first positional argument is a path to a config (file or dir),
    return it and strip from overrides. Otherwise return args unchanged.
    """
    if not args:
        return args, None
    first = args[0]
    # Likely an override, not a path.
    if first.startswith("--") or "=" in first or first.startswith("+"):
        return args, None
    candidate = Path(first)
    if candidate.is_file() and candidate.name == "config.yaml":
        return args[1:], candidate
    if candidate.is_dir():
        cfg = candidate / "config.yaml"
        if cfg.is_file():
            return args[1:], cfg
    return args, None


@click.group(help="landscapy-ml CLI (training utilities).")
def cli() -> None:
    pass


@cli.command("list", help="List registered models and data builders.")
def list_registered() -> None:
    import_builtin_examples()
    click.echo("Models:")
    for name in sorted(_MODEL_REGISTRY):
        click.echo(f"  - {name}")
    click.echo("\nData builders:")
    for name in sorted(_DATA_REGISTRY):
        click.echo(f"  - {name}")
    click.echo("\nLandscape regression runners:")
    for name in available_landscape_regression_runners():
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
    extracted_cfg: Path | None = None
    if config_path is None:
        overrides, extracted_cfg = _extract_config_path_from_overrides(overrides)
    cfg_path = config_path or extracted_cfg
    if cfg_path:
        cfg_path = cfg_path.resolve()
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
@click.option(
    "--csv-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--sequence-column", required=True, help="Column name containing raw sequences."
)
@click.option(
    "--label-column", required=True, help="Column name containing class labels."
)
@click.option(
    "--out-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=Path("landscapyml_run_conf"),
    show_default=True,
)
@click.option(
    "--embedding-mode",
    default="hard",
    show_default=True,
    help="Embedding mode: hard or soft.",
)
@click.option(
    "--model-name",
    default="facebook/esm2_t6_8M_UR50D",
    show_default=True,
    help="PLM model for embeddings.",
)
@click.option("--max-epochs", default=5, show_default=True, help="Max training epochs.")
@click.option(
    "--use-wandb/--no-wandb",
    default=True,
    show_default=True,
    help="Enable Weights & Biases logging.",
)
@click.option(
    "--wandb-project", default=None, help="W&B project name (None = use wandb default)."
)
@click.option("--wandb-run-name", default=None, help="W&B run name.")
@click.option("--seed", default=None, type=int, help="Optional seed.")
@click.option(
    "--val-split",
    default=0.1,
    show_default=True,
    type=float,
    help="Fraction of training used for validation when no val data provided.",
)
@click.option("--val-seed", default=None, type=int, help="Optional seed for val split.")
@click.option(
    "--model-key",
    default="sequence_mlp_classifier",
    show_default=True,
    help="Registered model key to use (e.g., sequence_mlp_classifier, sequence_mlp_ensemble).",
)
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


def _json_object(raw: str | None, *, option_name: str) -> dict:
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"{option_name} must be a valid JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise click.ClickException(f"{option_name} must decode to a JSON object.")
    return value


@cli.command(
    "train-landscape",
    help=(
        "Train a landscape regression model on CSV split files and write JSON "
        "metrics next to each input CSV."
    ),
)
@click.option(
    "--csv-path",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input CSV or CSV.GZ file. Can be supplied multiple times.",
)
@click.option(
    "--demo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root containing dataset/split/*.csv(.gz) files.",
)
@click.option(
    "--model-key",
    required=True,
    help=(
        "Registered model key or landscape runner key, e.g. "
        "graph_attention_regressor or diffusion_prior_gp."
    ),
)
@click.option(
    "--data-name",
    default="landscape_graph_regression",
    show_default=True,
    help="Registered data builder for Lightning-backed landscape models.",
)
@click.option("--sequence-column", default="sequence", show_default=True)
@click.option("--target-column", default="target", show_default=True)
@click.option("--split-column", default="set", show_default=True)
@click.option("--validation-column", default="validation", show_default=True)
@click.option("--train-label", default="train", show_default=True)
@click.option("--test-label", default="test", show_default=True)
@click.option("--output-suffix", default="results", show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--max-epochs", type=int, default=50, show_default=True)
@click.option("--accelerator", default="auto", show_default=True)
@click.option("--devices", default="1", show_default=True)
@click.option(
    "--model-kwargs",
    default="{}",
    show_default=True,
    help="JSON object forwarded to the registered model factory.",
)
@click.option(
    "--data-kwargs",
    default="{}",
    show_default=True,
    help="JSON object forwarded to the registered data builder.",
)
@click.option(
    "--trainer-kwargs",
    default="{}",
    show_default=True,
    help="JSON object forwarded to the Lightning trainer factory.",
)
@click.option(
    "--fit-kwargs",
    default="{}",
    show_default=True,
    help="JSON object forwarded to non-Lightning fit helpers such as diffusion_prior_gp.",
)
@click.option("--tokenizer", default=None, help="Optional tokenizer for graph models.")
@click.option("--moltype", default="protein", show_default=True)
@click.option("--continue-on-error/--stop-on-error", default=False, show_default=True)
def train_landscape(
    csv_path: tuple[Path, ...],
    demo_root: Path | None,
    model_key: str,
    data_name: str,
    sequence_column: str,
    target_column: str,
    split_column: str,
    validation_column: str,
    train_label: str,
    test_label: str,
    output_suffix: str,
    seed: int | None,
    max_epochs: int,
    accelerator: str,
    devices: str,
    model_kwargs: str,
    data_kwargs: str,
    trainer_kwargs: str,
    fit_kwargs: str,
    tokenizer: str | None,
    moltype: str | None,
    continue_on_error: bool,
) -> None:
    parsed_devices: int | str
    try:
        parsed_devices = int(devices)
    except ValueError:
        parsed_devices = devices

    config = LandscapeRegressionConfig(
        model_key=model_key,
        csv_paths=csv_path,
        demo_root=demo_root,
        sequence_column=sequence_column,
        target_column=target_column,
        split_column=split_column,
        validation_column=validation_column,
        train_label=train_label,
        test_label=test_label,
        data_name=data_name,
        output_suffix=output_suffix,
        seed=seed,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=parsed_devices,
        model_kwargs=_json_object(model_kwargs, option_name="--model-kwargs"),
        data_kwargs=_json_object(data_kwargs, option_name="--data-kwargs"),
        trainer_kwargs=_json_object(trainer_kwargs, option_name="--trainer-kwargs"),
        fit_kwargs=_json_object(fit_kwargs, option_name="--fit-kwargs"),
        tokenizer=tokenizer,
        moltype=moltype,
        continue_on_error=continue_on_error,
    )
    results = run_landscape_regression(config)
    for result in results:
        status = result.get("status", "unknown")
        output_path = result.get("output_path")
        click.echo(f"{status}: {output_path}")


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
