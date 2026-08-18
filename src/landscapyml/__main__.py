"""Command-line interface for registered landscape regression workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import click

from .core.model_registry import _DATA_REGISTRY, _MODEL_REGISTRY
from .landscape_regression import (
    LandscapeRegressionConfig,
    available_landscape_regression_runners,
    import_builtin_examples,
    run_landscape_regression,
)


@click.group(help="landscapy-ml CLI (training utilities).")
def cli() -> None:
    """Expose landscapy-ml training and registry commands."""
    pass


@cli.command("list", help="List registered models and data builders.")
def list_registered() -> None:
    """Print registered models, data builders, and regression runners."""
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
    """Train registered landscape models for one or more CSV datasets.

    Parameters
    ----------
    csv_path : tuple of pathlib.Path
        Explicit CSV or compressed CSV inputs.
    demo_root : pathlib.Path or None
        Root searched for bundled dataset split files when ``csv_path`` is
        empty.
    model_key : str
        Registered model or landscape-regression runner name.
    data_name : str
        Registered data-builder name for Lightning-backed models.
    sequence_column : str
        CSV column containing sequences.
    target_column : str
        CSV column containing numeric targets.
    split_column : str
        CSV column containing train and test labels.
    validation_column : str
        CSV column identifying validation rows within the training split.
    train_label : str
        Value in ``split_column`` that identifies training rows.
    test_label : str
        Value in ``split_column`` that identifies test rows.
    output_suffix : str
        Suffix used for JSON result files.
    seed : int or None
        Optional random seed passed to the regression workflow.
    max_epochs : int
        Maximum Lightning training epochs.
    accelerator : str
        Lightning accelerator selection.
    devices : str
        Lightning device count or device selection string.
    model_kwargs : str
        JSON object forwarded to the model factory.
    data_kwargs : str
        JSON object forwarded to the data builder.
    trainer_kwargs : str
        JSON object forwarded to the trainer factory.
    fit_kwargs : str
        JSON object forwarded to non-Lightning fit helpers.
    tokenizer : str or None
        Optional tokenizer identifier for graph inputs.
    moltype : str or None
        Sequence molecular type passed to Landscapy.
    continue_on_error : bool
        Continue processing later CSV files after an input fails.

    Returns
    -------
    None
        Results are emitted through Click and written by the selected runner.

    Raises
    ------
    click.ClickException
        If a JSON option does not decode to an object.
    """
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
    """Run the command-line interface without forcing process termination.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments. ``None`` delegates argument discovery to Click.

    Returns
    -------
    int
        Process-style exit status reported by Click.
    """
    try:
        cli.main(args=argv, prog_name="landscapyml", standalone_mode=False)
    except SystemExit as exc:  # click exits via SystemExit
        return exc.code
    except Exception:
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
