from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from ..core._optional import is_missing_optional_dependency
from ..core.adaptor import NodeIndexInputAdapter, register_input_adapter
from ..core.inference import infer_fitness_layer_from_landscape

try:
    import gpytorch  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if not is_missing_optional_dependency(exc, "gpytorch"):
        raise
    gpytorch = None  # type: ignore
    _GPYTORCH_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _GPYTORCH_IMPORT_ERROR = None

try:
    from fitness_landscape.analysis.diffusion_scale import (
        compute_ruggedness_diffusion_scale,
    )
    from fitness_landscape.core.fitness import NumericFitness
    from fitness_landscape.core.landscape import FitnessLandscape
    from fitness_landscape.transforms.eigenmode import eigenmode_decomposition
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if not is_missing_optional_dependency(exc, "fitness_landscape"):
        raise
    compute_ruggedness_diffusion_scale = None  # type: ignore
    NumericFitness = Any  # type: ignore
    FitnessLandscape = Any  # type: ignore
    eigenmode_decomposition = None  # type: ignore
    _LANDSCAPY_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _LANDSCAPY_IMPORT_ERROR = None


DEFAULT_MASK_TOKENS: tuple[str, ...] = ("X", "?", "-", "gap")


def _load_gpytorch():
    if gpytorch is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "GP fitness examples require gpytorch to be installed."
        ) from _GPYTORCH_IMPORT_ERROR
    return gpytorch


def _require_landscapy_bits() -> None:
    if compute_ruggedness_diffusion_scale is None or eigenmode_decomposition is None:
        raise ImportError(
            "This example requires landscapy analysis utilities to be importable."
        ) from _LANDSCAPY_IMPORT_ERROR


def _aggregate_numeric_targets(
    layer: Any, *, aggregate_func: Optional[Callable[..., Any]] = np.mean
) -> np.ndarray:
    if getattr(layer, "dtype", None) != "numeric":
        raise ValueError(
            "The GP example currently supports numeric fitness layers only."
        )
    if aggregate_func is None:
        values = layer.to_scalar()
    else:
        try:
            values = layer.to_scalar(aggregate_func=aggregate_func)
        except TypeError:
            values = layer.to_scalar()
    return np.asarray(values, dtype=float).reshape(-1)


def _normalize_mask_tokens(mask_tokens: Sequence[str] | None) -> tuple[str, ...]:
    if mask_tokens is None:
        return ()
    return tuple(str(token) for token in mask_tokens)


def sequence_has_masked_residue(
    sequence: Any,
    *,
    mask_tokens: Sequence[str] = DEFAULT_MASK_TOKENS,
) -> bool:
    """
    Return ``True`` when a sequence contains any configured masking token.
    """

    tokens = set(_normalize_mask_tokens(mask_tokens))
    if not tokens:
        return False

    if hasattr(sequence, "to_array"):
        arr = np.asarray(sequence.to_array(), dtype=object).reshape(-1)
    elif isinstance(sequence, str):
        arr = np.asarray(list(sequence), dtype=object)
    else:
        arr = np.asarray(sequence, dtype=object).reshape(-1)
    return any(str(value) in tokens for value in arr)


def _subset_numeric_layer(layer: Any, indices: Sequence[int], *, name: str) -> Any:
    values: list[list[float]] = []
    for idx in indices:
        raw_value = layer.get_value(int(idx))
        arr = np.asarray(raw_value, dtype=float).reshape(-1)
        if arr.size == 0:
            arr = np.asarray([float("nan")], dtype=float)
        values.append(arr.tolist())
    metadata = dict(getattr(layer, "metadata", {}) or {})
    return NumericFitness(name=name, values=values, metadata=metadata)


def _build_landscape_subset(
    landscape: Any,
    indices: Sequence[int],
    *,
    target_layer: str,
) -> Any:
    if NumericFitness is Any or FitnessLandscape is Any:  # pragma: no cover - optional
        raise ImportError(
            "Constructing diffusion-scale subsets requires landscapy to be importable."
        )

    seq_indices = [int(idx) for idx in indices]
    if not seq_indices:
        raise ValueError("Cannot build a subset landscape with zero nodes.")

    if not hasattr(landscape, "_node_order"):
        raise ValueError(
            "Landscape subset construction requires an index-aligned graph node order."
        )

    selected_nodes = [landscape._node_order[idx] for idx in seq_indices]
    graph = landscape.graph.subgraph(selected_nodes).copy()
    sequences = [landscape.sequences[idx] for idx in seq_indices]
    layer = landscape.fitness_layers[target_layer]
    fitness_layers = {
        target_layer: _subset_numeric_layer(layer, seq_indices, name=target_layer)
    }

    embeddings = None
    if getattr(landscape, "embeddings", None):
        embeddings = {
            domain: np.asarray(arr)[seq_indices].copy()
            for domain, arr in landscape.embeddings.items()
        }

    subset = FitnessLandscape(
        sequences=sequences,
        graph=graph,
        fitness_layers=fitness_layers,
        embeddings=embeddings,
        emb_arr_key=getattr(landscape, "_emb_arr_key", "emb_arr"),
        active_embedding_domain=getattr(landscape, "_active_embedding_domain", None),
        embedding_metadata=getattr(landscape, "_embedding_metadata", None),
    )
    subset.view(target_layer)
    return subset


def _compute_diffusion_covariance_matrix(
    graph: Any,
    *,
    t: float,
    signal_variance: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    _require_landscapy_bits()
    eigenvalues, eigenvectors = eigenmode_decomposition(  # type: ignore[misc]
        graph,
        matrix="norm_laplacian",
    )
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    lambda_adjusted = eigenvalues + float(epsilon)
    heat = np.exp(-float(t) * lambda_adjusted)
    scale = (float(signal_variance) * len(eigenvalues)) / max(float(np.sum(heat)), epsilon)
    cov = eigenvectors @ np.diag(heat * scale) @ eigenvectors.T
    return 0.5 * (cov + cov.T)


def _scalar_feature_normalization_stats(
    values: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float32).view(-1)
    if tensor.numel() == 0:
        raise ValueError("Cannot normalize an empty feature tensor.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Cannot normalize features containing NaN or Inf values.")
    mean = float(tensor.mean().item())
    scale = float(tensor.std(unbiased=False).item())
    if scale <= float(eps):
        scale = 1.0
    return mean, scale


@dataclass
class DiffusionGPArtifacts:
    covariance_matrix: torch.Tensor
    all_inputs: torch.Tensor
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    full_targets: torch.Tensor
    observed_mask: torch.Tensor
    predict_mask: torch.Tensor
    fit_indices: torch.Tensor
    train_indices: torch.Tensor
    t_map: float
    signal_variance: float
    mask_tokens: tuple[str, ...]
    normalize_features: bool = False
    feature_normalization_mean: float = 0.0
    feature_normalization_scale: float = 1.0


@dataclass
class DiffusionGPFitResult:
    model: "DiffusionPriorExactGP"
    likelihood: Any
    artifacts: DiffusionGPArtifacts
    losses: list[float]


def build_diffusion_gp_artifacts_from_landscape(
    landscape: Any,
    *,
    target_layer: str,
    aggregate_func: Optional[Callable[..., Any]] = np.mean,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    mask_tokens: Sequence[str] = DEFAULT_MASK_TOKENS,
    drop_masked_sequences_for_t_map: bool = True,
    t_min: float = 0.01,
    t_max: float = 100.0,
    epsilon: float = 1e-8,
    method: str = "grid",
    grid_size: int = 256,
    prior: str = "log_uniform",
    bootstrap_samples: int = 200,
    random_state: Optional[int] = None,
    min_signal_variance: float = 1e-6,
    normalize_features: bool = False,
    feature_normalization_eps: float = 1e-8,
) -> DiffusionGPArtifacts:
    """
    Build the fixed diffusion covariance and node-index training inputs for an
    exact GP over a single landscape graph.

    ``t_MAP`` is fit on the training, unmasked subgraph when possible, then the
    resulting diffusion scale is reused to construct a covariance over the full
    landscape graph, including held-out and masked sequences. If
    ``train_indices`` is omitted, all finite targets are treated as training
    observations.
    """

    _require_landscapy_bits()

    if getattr(landscape, "graph", None) is None:
        raise ValueError("Landscape must have a graph before fitting the GP example.")
    layers = getattr(landscape, "fitness_layers", None)
    if not isinstance(layers, Mapping) or target_layer not in layers:
        raise ValueError(f"Landscape does not contain target layer '{target_layer}'.")

    target_values = _aggregate_numeric_targets(
        layers[target_layer],
        aggregate_func=aggregate_func,
    )
    full_targets = torch.as_tensor(target_values, dtype=torch.float32)
    if full_targets.ndim != 1:
        full_targets = full_targets.view(-1)
    if full_targets.shape[0] != len(landscape.sequences):
        raise ValueError("Target layer length does not match the number of landscape sequences.")

    num_nodes = full_targets.shape[0]
    observed_mask = torch.isfinite(full_targets)
    if train_indices is None:
        resolved_train_indices = torch.nonzero(observed_mask, as_tuple=False).view(-1)
    else:
        resolved_train_indices = torch.as_tensor(train_indices, dtype=torch.long).view(-1)
        if resolved_train_indices.numel() > 0:
            if bool((resolved_train_indices < 0).any()) or bool(
                (resolved_train_indices >= num_nodes).any()
            ):
                raise ValueError(
                    f"train_indices contains node indices outside [0, {num_nodes})."
                )
            resolved_train_indices = torch.unique(resolved_train_indices, sorted=True)
            if not bool(observed_mask[resolved_train_indices].all()):
                raise ValueError(
                    "train_indices contains nodes without finite target values."
                )
    if resolved_train_indices.numel() < 2:
        raise ValueError(
            "At least two observed numeric fitness values are required for the GP example."
        )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[resolved_train_indices] = True

    normalized_mask_tokens = _normalize_mask_tokens(mask_tokens)
    if drop_masked_sequences_for_t_map:
        clean_sequence_mask = np.asarray(
            [
                not sequence_has_masked_residue(
                    sequence,
                    mask_tokens=normalized_mask_tokens,
                )
                for sequence in landscape.sequences
            ],
            dtype=bool,
        )
    else:
        clean_sequence_mask = np.ones(num_nodes, dtype=bool)

    train_np = train_mask.detach().cpu().numpy()
    fit_mask = clean_sequence_mask & train_np
    if int(np.sum(fit_mask)) < 2:
        if drop_masked_sequences_for_t_map:
            warnings.warn(
                "Fewer than two observed, fully unmasked nodes were available for "
                "diffusion-scale fitting; falling back to all training nodes.",
                RuntimeWarning,
            )
            fit_mask = train_np
        if int(np.sum(fit_mask)) < 2:
            raise ValueError(
                "Unable to identify enough observed nodes to estimate the diffusion scale."
            )

    fit_indices_np = np.flatnonzero(fit_mask)
    fit_subset = _build_landscape_subset(
        landscape,
        fit_indices_np.tolist(),
        target_layer=target_layer,
    )
    fit_result = compute_ruggedness_diffusion_scale(  # type: ignore[misc]
        fit_subset,
        t_min=t_min,
        t_max=t_max,
        epsilon=epsilon,
        method=method,
        grid_size=grid_size,
        prior=prior,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    t_map = float(fit_result["t_map"])

    fit_values = target_values[fit_indices_np]
    centered = fit_values - float(np.mean(fit_values))
    signal_variance = float(np.var(centered, ddof=1)) if centered.size > 1 else 0.0
    signal_variance = max(signal_variance, float(min_signal_variance))

    covariance = _compute_diffusion_covariance_matrix(
        landscape.graph,
        t=t_map,
        signal_variance=signal_variance,
        epsilon=epsilon,
    )
    covariance_tensor = torch.as_tensor(covariance, dtype=torch.float32)

    all_inputs = torch.arange(num_nodes, dtype=torch.float32).view(-1, 1)
    train_inputs = all_inputs[resolved_train_indices]
    train_targets = full_targets[resolved_train_indices]
    feature_mean = 0.0
    feature_scale = 1.0
    if normalize_features:
        feature_mean, feature_scale = _scalar_feature_normalization_stats(
            train_inputs,
            eps=feature_normalization_eps,
        )

    return DiffusionGPArtifacts(
        covariance_matrix=covariance_tensor,
        all_inputs=all_inputs,
        train_inputs=train_inputs,
        train_targets=train_targets,
        full_targets=full_targets,
        observed_mask=observed_mask,
        predict_mask=~train_mask,
        fit_indices=torch.as_tensor(fit_indices_np, dtype=torch.long),
        train_indices=resolved_train_indices.to(torch.long),
        t_map=t_map,
        signal_variance=signal_variance,
        mask_tokens=normalized_mask_tokens,
        normalize_features=bool(normalize_features),
        feature_normalization_mean=feature_mean,
        feature_normalization_scale=feature_scale,
    )


if gpytorch is not None:  # pragma: no branch - import-time optional dependency
    _KernelBase = gpytorch.kernels.Kernel
    _ExactGPBase = gpytorch.models.ExactGP
else:  # pragma: no cover - exercised only without gpytorch installed
    _KernelBase = torch.nn.Module
    _ExactGPBase = torch.nn.Module


class DiffusionPriorKernel(_KernelBase):
    """
    GPyTorch kernel backed by a fixed, precomputed landscape covariance matrix.
    """

    is_stationary = False

    def __init__(
        self,
        covariance_matrix: torch.Tensor,
        *,
        jitter: float = 1e-4,
        normalize_features: bool = False,
        feature_normalization_mean: float = 0.0,
        feature_normalization_scale: float = 1.0,
    ) -> None:
        _load_gpytorch()
        super().__init__()
        cov = torch.as_tensor(covariance_matrix, dtype=torch.float32)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance_matrix must be a square matrix.")
        self.register_buffer("covariance_matrix", cov)
        self.register_buffer(
            "feature_normalization_mean",
            torch.tensor(float(feature_normalization_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "feature_normalization_scale",
            torch.tensor(float(feature_normalization_scale), dtype=torch.float32),
        )
        self.normalize_features = bool(normalize_features)
        self.jitter = float(jitter)
        if self.normalize_features and float(feature_normalization_scale) <= 0:
            raise ValueError("feature_normalization_scale must be positive.")

    def _to_index_tensor(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1).to(dtype=torch.float32)
        if self.normalize_features:
            scale = self.feature_normalization_scale.to(device=flat.device, dtype=flat.dtype)
            mean = self.feature_normalization_mean.to(device=flat.device, dtype=flat.dtype)
            flat = flat * scale + mean
        flat = torch.round(flat).to(dtype=torch.long)
        if flat.numel() == 0:
            raise ValueError("GP inputs must contain at least one node index.")
        if bool((flat < 0).any()) or bool((flat >= self.covariance_matrix.shape[0]).any()):
            raise ValueError("GP node-index inputs are outside the covariance matrix.")
        return flat

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        diag: bool = False,
        **params: Any,
    ):
        del params
        if x2 is None:
            x2 = x1
        idx1 = self._to_index_tensor(x1)
        idx2 = self._to_index_tensor(x2)
        cov = self.covariance_matrix.index_select(0, idx1).index_select(1, idx2)

        if idx1.numel() == idx2.numel() and torch.equal(idx1, idx2) and self.jitter > 0:
            cov = cov + torch.eye(
                cov.shape[0],
                dtype=cov.dtype,
                device=cov.device,
            ) * self.jitter

        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


class DiffusionPriorExactGP(_ExactGPBase):
    """
    Exact GP regressor whose covariance is induced by landscape diffusion scale.

    This is intentionally a transductive model over a fixed graph: inputs are
    node indices, and the graph-derived covariance encodes the prior over all
    sequences already present in the landscape.
    """

    layer_kind = "numeric"

    def __init__(
        self,
        *,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        covariance_matrix: torch.Tensor,
        likelihood: Any = None,
        mean_mode: str = "constant",
        jitter: float = 1e-4,
        normalize_features: bool = False,
        feature_normalization_mean: float = 0.0,
        feature_normalization_scale: float = 1.0,
    ) -> None:
        gp = _load_gpytorch()
        if likelihood is None:
            likelihood = gp.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.normalize_features = bool(normalize_features)
        self.register_buffer(
            "feature_normalization_mean",
            torch.tensor(float(feature_normalization_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "feature_normalization_scale",
            torch.tensor(float(feature_normalization_scale), dtype=torch.float32),
        )
        if self.normalize_features and float(feature_normalization_scale) <= 0:
            raise ValueError("feature_normalization_scale must be positive.")
        if mean_mode == "constant":
            self.mean_module = gp.means.ConstantMean()
        elif mean_mode == "zero":
            self.mean_module = gp.means.ZeroMean()
        else:
            raise ValueError("mean_mode must be either 'constant' or 'zero'.")
        self.covar_module = DiffusionPriorKernel(
            covariance_matrix=covariance_matrix,
            jitter=jitter,
            normalize_features=normalize_features,
            feature_normalization_mean=feature_normalization_mean,
            feature_normalization_scale=feature_normalization_scale,
        )

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return x
        scale = self.feature_normalization_scale.to(device=x.device, dtype=x.dtype)
        mean = self.feature_normalization_mean.to(device=x.device, dtype=x.dtype)
        return (x - mean) / scale

    def forward(self, x: torch.Tensor):
        gp = _load_gpytorch()
        normalized_x = self._normalize_inputs(x)
        mean_x = self.mean_module(normalized_x)
        covar_x = self.covar_module(normalized_x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_distribution(self, inputs: torch.Tensor):
        gp = _load_gpytorch()
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            return self.likelihood(self(inputs))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        posterior = self.predict_distribution(inputs)
        return posterior.mean


class LandscapeNodeIndexInputAdapter(NodeIndexInputAdapter):
    name = "landscape_node_index"


def fit_diffusion_prior_gp(
    landscape: Any,
    *,
    target_layer: str,
    aggregate_func: Optional[Callable[..., Any]] = np.mean,
    train_indices: Sequence[int] | torch.Tensor | None = None,
    mask_tokens: Sequence[str] = DEFAULT_MASK_TOKENS,
    drop_masked_sequences_for_t_map: bool = True,
    t_min: float = 0.01,
    t_max: float = 100.0,
    epsilon: float = 1e-8,
    method: str = "grid",
    grid_size: int = 256,
    prior: str = "log_uniform",
    bootstrap_samples: int = 200,
    random_state: Optional[int] = None,
    min_signal_variance: float = 1e-6,
    normalize_features: bool = False,
    feature_normalization_eps: float = 1e-8,
    training_iters: int = 100,
    learning_rate: float = 0.1,
    jitter: float = 1e-4,
    mean_mode: str = "constant",
    device: Optional[str | torch.device] = None,
) -> DiffusionGPFitResult:
    """
    Fit an exact diffusion-prior GP on observed node fitness values.

    This example is intended for small to medium landscapes where exact GP
    inference remains tractable. If ``train_indices`` is supplied, held-out
    finite targets remain available for evaluation but are not used for
    diffusion-scale fitting or GP likelihood optimization.
    """

    gp = _load_gpytorch()
    artifacts = build_diffusion_gp_artifacts_from_landscape(
        landscape,
        target_layer=target_layer,
        aggregate_func=aggregate_func,
        train_indices=train_indices,
        mask_tokens=mask_tokens,
        drop_masked_sequences_for_t_map=drop_masked_sequences_for_t_map,
        t_min=t_min,
        t_max=t_max,
        epsilon=epsilon,
        method=method,
        grid_size=grid_size,
        prior=prior,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
        min_signal_variance=min_signal_variance,
        normalize_features=normalize_features,
        feature_normalization_eps=feature_normalization_eps,
    )

    if artifacts.train_inputs.shape[0] > 2000:
        warnings.warn(
            "DiffusionPriorExactGP uses exact GP inference and may become slow or "
            "memory-intensive for more than a few thousand observed nodes.",
            RuntimeWarning,
        )

    training_device = torch.device(device) if device is not None else torch.device("cpu")
    train_x = artifacts.train_inputs.to(training_device)
    train_y = artifacts.train_targets.to(training_device)
    covariance_matrix = artifacts.covariance_matrix.to(training_device)

    model = DiffusionPriorExactGP(
        train_x=train_x,
        train_y=train_y,
        covariance_matrix=covariance_matrix,
        mean_mode=mean_mode,
        jitter=jitter,
        normalize_features=artifacts.normalize_features,
        feature_normalization_mean=artifacts.feature_normalization_mean,
        feature_normalization_scale=artifacts.feature_normalization_scale,
    )
    model = model.to(training_device)
    model.likelihood = model.likelihood.to(training_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    marginal_log_likelihood = gp.mlls.ExactMarginalLogLikelihood(
        model.likelihood,
        model,
    )

    losses: list[float] = []
    model.train()
    model.likelihood.train()
    for _ in range(int(training_iters)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    model.eval()
    model.likelihood.eval()
    return DiffusionGPFitResult(
        model=model,
        likelihood=model.likelihood,
        artifacts=artifacts,
        losses=losses,
    )


def attach_diffusion_gp_predictions(
    landscape: Any,
    model: DiffusionPriorExactGP,
    *,
    layer_name: str = "diffusion_gp_predicted_fitness",
    attach: bool = True,
    inplace: bool = True,
) -> Any:
    """
    Predict fitness at every landscape node and attach the mean as a numeric layer.
    """

    return infer_fitness_layer_from_landscape(
        landscape,
        model,
        batch_size=256,
        attach=attach,
        inplace=inplace,
        layer_name=layer_name,
        input_adapter=NodeIndexInputAdapter.name,
    )


register_input_adapter(
    LandscapeNodeIndexInputAdapter.name,
    LandscapeNodeIndexInputAdapter,
    overwrite=True,
)


__all__ = [
    "DEFAULT_MASK_TOKENS",
    "DiffusionGPArtifacts",
    "DiffusionGPFitResult",
    "DiffusionPriorExactGP",
    "DiffusionPriorKernel",
    "LandscapeNodeIndexInputAdapter",
    "attach_diffusion_gp_predictions",
    "build_diffusion_gp_artifacts_from_landscape",
    "fit_diffusion_prior_gp",
    "sequence_has_masked_residue",
]
