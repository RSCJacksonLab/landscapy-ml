from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from ..core.adaptor import (
    LandscapeInputAdapter,
    ModelAdapter,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
)


def _import_numeric_fitness():
    try:
        from fitness_landscape.core.fitness import NumericFitness  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NumericFitness is required to attach Boltz2 outputs to a landscape. "
            "Install fitness_landscape to enable numeric layer construction."
        ) from exc
    return NumericFitness


def _numeric_output_adapter(
    outputs: Mapping[str, torch.Tensor],
    categories: Optional[Iterable[str]],  # noqa: ARG001 - numeric layers ignore categories
    metadata: Mapping[str, Any],
    layer_name: str,
):
    tensor = outputs.get("output")
    if tensor is None and len(outputs) == 1:
        tensor = next(iter(outputs.values()))
    if tensor is None:
        raise ValueError(
            "Numeric adapter expects a single tensor output or an 'output' key."
        )
    if not torch.is_tensor(tensor):
        raise TypeError("Numeric adapter received non-tensor output.")

    NumericFitness = _import_numeric_fitness()
    meta = dict(metadata)
    meta.setdefault("layer_kind", "numeric")
    cpu_tensor = tensor.detach().cpu()
    if hasattr(NumericFitness, "from_tensor"):
        return NumericFitness.from_tensor(
            name=layer_name,
            tensor=cpu_tensor,
            metadata=meta,
        )

    values = cpu_tensor.numpy()
    if values.ndim == 1:
        values = values[:, None]
    return NumericFitness(name=layer_name, values=values.tolist(), metadata=meta)


register_layer_adapter("numeric", _numeric_output_adapter, overwrite=True)


class Boltz2InputAdapter(LandscapeInputAdapter):
    """
    Example bridge from ``FitnessLandscape`` objects to Boltz2 dataloaders.

    The adapter is intentionally lightweight: it relies on a caller-supplied
    builder or a landscape-provided ``get_boltz2_datamodule`` hook instead of
    hardcoding project-specific preprocessing rules into the package core.
    """

    name = "boltz2"

    def __init__(
        self,
        datamodule_builder: Optional[Callable[[Any], Any]] = None,
        dataloader_builder: Optional[Callable[[Any], Iterable[Any]]] = None,
    ) -> None:
        self.datamodule_builder = datamodule_builder
        self.dataloader_builder = dataloader_builder

    def metadata(self, landscape: Any) -> Mapping[str, Any]:
        return {"input_adapter": self.name, "example_integration": "boltz2"}

    def embedding_info(self, landscape: Any) -> tuple[None, None, None]:  # noqa: ARG002
        return None, None, None

    def _get_dataloader(self, landscape: Any) -> Iterable[Any]:
        if self.dataloader_builder is not None:
            return self.dataloader_builder(landscape)

        dm = None
        if self.datamodule_builder is not None:
            dm = self.datamodule_builder(landscape)
        elif hasattr(landscape, "get_boltz2_datamodule"):
            dm = landscape.get_boltz2_datamodule()

        if dm is None:
            raise RuntimeError(
                "Boltz2InputAdapter requires a datamodule_builder or dataloader_builder, "
                "or a landscape method get_boltz2_datamodule()."
            )
        if hasattr(dm, "predict_dataloader"):
            return dm.predict_dataloader()
        if hasattr(dm, "test_dataloader"):
            return dm.test_dataloader()
        if hasattr(dm, "val_dataloader"):
            return dm.val_dataloader()
        raise RuntimeError("Boltz2 datamodule does not expose a usable dataloader.")

    def iter_batches(
        self,
        landscape: Any,
        *,
        batch_size: int,  # noqa: ARG002 - Boltz loaders are preconfigured
        num_workers: int = 0,  # noqa: ARG002 - Boltz loaders are preconfigured
        device: Optional[torch.device] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterable[Any]:
        for batch in self._get_dataloader(landscape):
            yield batch

    def to_model_inputs(
        self, batch: Any, *, device: Optional[torch.device] = None
    ) -> Any:
        if device is None or not isinstance(batch, Mapping):
            return batch

        # Mirror Boltz transfer rules: some bookkeeping stays on CPU.
        keep_on_cpu = {
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        }
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and key not in keep_on_cpu:
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved


class Boltz2ModelAdapter(ModelAdapter):
    """
    Example adapter for ``boltz.model.models.boltz2.Boltz2`` outputs.
    """

    layer_kind = "numeric"

    def __init__(
        self,
        model: Any,
        *,
        output_key: str = "affinity_pred_value",
        probability_key: Optional[str] = "affinity_probability_binary",
    ) -> None:
        self.model = model
        self.output_key = output_key
        self.probability_key = probability_key

    def eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor]:
        if hasattr(self.model, "predict_step"):
            out = self.model.predict_step(inputs, batch_idx=0, dataloader_idx=0)  # type: ignore[arg-type]
        else:
            out = self.model(inputs)

        if out is None:
            return {}
        if isinstance(out, Mapping):
            if self.output_key and self.output_key in out:
                result = {"output": out[self.output_key]}
                if self.probability_key and self.probability_key in out:
                    result["probability"] = out[self.probability_key]
                return result
            return out
        if torch.is_tensor(out):
            return {"output": out}
        raise TypeError("Boltz2ModelAdapter received unsupported model output type.")


class Boltz2AffinityProbAdapter(ModelAdapter):
    """
    Example adapter that maps Boltz2 affinity probabilities to a binary
    probabilistic categorical layer.
    """

    layer_kind = "prob_categorical"

    def __init__(
        self,
        model: Any,
        *,
        probability_key: str = "affinity_probability_binary",
    ) -> None:
        self.model = model
        self.probability_key = probability_key

    def eval(self) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict(self, inputs: Any) -> Mapping[str, torch.Tensor]:
        if hasattr(self.model, "predict_step"):
            out = self.model.predict_step(inputs, batch_idx=0, dataloader_idx=0)  # type: ignore[arg-type]
        else:
            out = self.model(inputs)
        if not isinstance(out, Mapping):
            raise TypeError("Expected mapping output from Boltz2 for probabilistic affinity.")
        if self.probability_key not in out:
            raise ValueError(f"Missing '{self.probability_key}' in Boltz2 output.")
        probs = out[self.probability_key]
        if probs.ndim == 0:
            probs = probs[None]
        return {"mean": torch.stack([1.0 - probs, probs], dim=-1)}


def _maybe_register_boltz2_model() -> None:
    try:
        from boltz.model.models.boltz2 import Boltz2  # type: ignore
    except Exception:
        return

    register_model_layer_mapping(Boltz2, "numeric", overwrite=True)
    register_model_adapter(Boltz2, lambda model: Boltz2ModelAdapter(model), overwrite=True)


def _load_boltz2_components():
    try:
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # type: ignore
        from boltz.data.types import Manifest  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boltz2 components are not available. Install the boltz package to use "
            "the example adapter builders."
        ) from exc
    return Boltz2InferenceDataModule, Manifest


def build_boltz2_datamodule_from_processed(
    processed_dir: Path,
    *,
    num_workers: int = 0,
    override_method: Optional[str] = None,
    affinity: bool = False,
):
    """
    Example builder for a Boltz2 inference datamodule backed by a processed directory.
    """

    Boltz2InferenceDataModule, Manifest = _load_boltz2_components()
    processed_dir = Path(processed_dir)
    manifest = Manifest.load(processed_dir / "manifest.json")
    return Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=processed_dir / "mols",
        constraints_dir=processed_dir / "constraints",
        template_dir=processed_dir / "templates",
        extra_mols_dir=processed_dir / "mols",
        num_workers=num_workers,
        override_method=override_method,
        affinity=affinity,
    )


def make_boltz2_input_adapter_for_processed(
    processed_dir: Path,
    *,
    num_workers: int = 0,
    override_method: Optional[str] = None,
    affinity: bool = False,
) -> Boltz2InputAdapter:
    processed_dir = Path(processed_dir).resolve()

    def _builder(_landscape: Any):
        return build_boltz2_datamodule_from_processed(
            processed_dir,
            num_workers=num_workers,
            override_method=override_method,
            affinity=affinity,
        )

    return Boltz2InputAdapter(datamodule_builder=_builder)


register_input_adapter(Boltz2InputAdapter.name, lambda: Boltz2InputAdapter(), overwrite=True)
_maybe_register_boltz2_model()


__all__ = [
    "Boltz2AffinityProbAdapter",
    "Boltz2InputAdapter",
    "Boltz2ModelAdapter",
    "build_boltz2_datamodule_from_processed",
    "make_boltz2_input_adapter_for_processed",
]
