"""Diffusion-prior exact Gaussian-process example for Landscapy graphs."""

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
    """Test whether a sequence contains a configured masking token.

    Parameters
    ----------
    sequence : Any
        Landscapy sequence object, string, or array-like token sequence.
    mask_tokens : sequence of str, default=DEFAULT_MASK_TOKENS
        Tokens interpreted as masked, ambiguous, or gap residues.

    Returns
    -------
    bool
        ``True`` when at least one flattened token matches exactly.
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
    """Store fixed covariance, node inputs, targets, and fitted prior scale.

    Parameters
    ----------
    covariance_matrix : torch.Tensor
        CPU ``float32`` covariance with shape ``(n_nodes, n_nodes)``.
    all_inputs : torch.Tensor
        CPU ``float32`` node-index column with shape ``(n_nodes, 1)``.
    train_inputs : torch.Tensor
        Training rows selected from ``all_inputs``.
    train_targets : torch.Tensor
        Finite ``float32`` targets aligned with ``train_inputs``.
    full_targets : torch.Tensor
        All aligned targets, including non-finite held-out observations.
    observed_mask : torch.Tensor
        Boolean mask of finite source targets.
    predict_mask : torch.Tensor
        Boolean complement of selected training nodes.
    fit_indices : torch.Tensor
        Nodes used to estimate diffusion scale.
    train_indices : torch.Tensor
        Nodes used for GP likelihood optimization.
    t_map : float
        Maximum-a-posteriori diffusion scale.
    signal_variance : float
        Training signal variance used to scale covariance.
    mask_tokens : tuple of str
        Tokens excluded from diffusion-scale fitting when requested.
    normalize_features : bool, default=False
        Whether scalar node-index inputs are normalized in the model.
    feature_normalization_mean : float, default=0.0
        Training-index mean.
    feature_normalization_scale : float, default=1.0
        Positive training-index population standard deviation.
    """

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
    """Store a fitted diffusion GP and its reproducibility artifacts.

    Parameters
    ----------
    model : DiffusionPriorExactGP
        Fitted exact Gaussian-process model.
    likelihood : Any
        Fitted GPyTorch Gaussian likelihood.
    artifacts : DiffusionGPArtifacts
        Covariance, split, target, scale, and normalization inputs.
    losses : list of float
        Marginal negative log-likelihood after each optimization iteration.
    """

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
    """Build fixed diffusion covariance and node-index GP inputs.

    ``t_MAP`` is fit on the training, unmasked subgraph when possible, then the
    resulting diffusion scale is reused to construct a covariance over the full
    landscape graph, including held-out and masked sequences. If
    ``train_indices`` is omitted, all finite targets are treated as training
    observations.

    Parameters
    ----------
    landscape : Any
        Landscapy fitness landscape with a graph, sequences, and fitness layers.
    target_layer : str
        Numeric fitness-layer name.
    aggregate_func : callable or None, default=numpy.mean
        Replicate aggregation function.
    train_indices : sequence of int, torch.Tensor, or None, optional
        Canonical nodes used for GP training. Defaults to all finite targets.
    mask_tokens : sequence of str, default=DEFAULT_MASK_TOKENS
        Tokens treated as masked during diffusion-scale estimation.
    drop_masked_sequences_for_t_map : bool, default=True
        Exclude masked training sequences from diffusion-scale fitting when at
        least two unmasked training nodes remain.
    t_min : float, default=0.01
        Minimum diffusion scale passed to Landscapy estimation.
    t_max : float, default=100.0
        Maximum diffusion scale passed to Landscapy estimation.
    epsilon : float, default=1e-8
        Positive numerical adjustment for diffusion estimation and covariance.
    method : str, default="grid"
        Landscapy diffusion-scale optimization method.
    grid_size : int, default=256
        Candidate count for grid optimization.
    prior : str, default="log_uniform"
        Diffusion-scale prior used by Landscapy.
    bootstrap_samples : int, default=200
        Bootstrap replicates used by Landscapy estimation.
    random_state : int or None, optional
        Random seed for diffusion-scale estimation.
    min_signal_variance : float, default=1e-6
        Lower bound for covariance signal variance.
    normalize_features : bool, default=False
        Record scalar node-index normalization for the GP model.
    feature_normalization_eps : float, default=1e-8
        Minimum population standard deviation before replacement by one.

    Returns
    -------
    DiffusionGPArtifacts
        CPU tensors and scalar settings required to construct and audit the GP.

    Raises
    ------
    ImportError
        If required Landscapy analysis functionality is unavailable.
    ValueError
        If the graph, target layer, target alignment, training indices, or
        eligible diffusion-scale subset is invalid.

    Notes
    -----
    Diffusion scale is a graph-dependent scientific prior. The fitted subset,
    mask policy, graph, scale bounds, prior, and random state should be retained
    with experimental outputs. Mask filtering falls back to all training nodes
    with a warning when fewer than two eligible unmasked nodes remain.
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
    """Index a fixed, precomputed landscape covariance matrix.

    Parameters
    ----------
    covariance_matrix : torch.Tensor
        Square ``float32`` covariance over all landscape nodes.
    jitter : float, default=1e-4
        Diagonal adjustment added when both index vectors are identical.
    normalize_features : bool, default=False
        Invert stored scalar normalization before converting inputs to indices.
    feature_normalization_mean : float, default=0.0
        Mean used to invert node-index normalization.
    feature_normalization_scale : float, default=1.0
        Positive scale used to invert node-index normalization.

    Attributes
    ----------
    covariance_matrix : torch.Tensor
        Registered covariance buffer, moved with the kernel device.
    jitter : float
        Diagonal numerical adjustment.
    normalize_features : bool
        Whether inputs require inverse normalization.

    Raises
    ------
    ImportError
        If GPyTorch is unavailable.
    ValueError
        If covariance is not square or an enabled normalization scale is not
        positive.

    Notes
    -----
    Inputs are transductive graph-node indices, not continuous coordinates.
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
        """Select covariance entries for two node-index batches.

        Parameters
        ----------
        x1 : torch.Tensor
            Node indices, optionally normalized, with arbitrary batch shape.
        x2 : torch.Tensor or None, optional
            Second node-index collection. Defaults to ``x1``.
        diag : bool, default=False
            Return only the covariance diagonal.
        **params : Any
            Ignored GPyTorch kernel compatibility arguments.

        Returns
        -------
        torch.Tensor
            Covariance matrix with shape ``(x1.numel(), x2.numel())`` or its
            diagonal when ``diag`` is true, on the covariance buffer device.

        Raises
        ------
        ValueError
            If an index collection is empty or contains nodes outside the
            covariance matrix.
        """
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
    """Regress fitness with a fixed landscape-diffusion covariance.

    This is intentionally a transductive model over a fixed graph: inputs are
    node indices, and the graph-derived covariance encodes the prior over all
    sequences already present in the landscape.

    Parameters
    ----------
    train_x : torch.Tensor
        Training node-index column on the intended model device.
    train_y : torch.Tensor
        One-dimensional finite training targets aligned with ``train_x``.
    covariance_matrix : torch.Tensor
        Square covariance over every node in the fixed landscape.
    likelihood : Any, optional
        GPyTorch likelihood. Defaults to ``GaussianLikelihood``.
    mean_mode : {"constant", "zero"}, default="constant"
        GP mean-function family.
    jitter : float, default=1e-4
        Diagonal kernel adjustment.
    normalize_features : bool, default=False
        Normalize scalar node-index inputs before evaluation.
    feature_normalization_mean : float, default=0.0
        Node-index normalization mean.
    feature_normalization_scale : float, default=1.0
        Positive node-index normalization scale.

    Attributes
    ----------
    likelihood : Any
        GPyTorch likelihood used for training and prediction.
    mean_module : Any
        Selected GPyTorch mean module.
    covar_module : DiffusionPriorKernel
        Fixed graph covariance kernel.
    layer_kind : str
        Output-adapter key ``"numeric"``.

    Raises
    ------
    ImportError
        If GPyTorch is unavailable.
    ValueError
        If mean mode is unknown or enabled normalization has a non-positive
        scale.
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
        """Construct the latent multivariate-normal distribution.

        Parameters
        ----------
        x : torch.Tensor
            Node-index column on the model device.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Latent GP distribution with fixed landscape covariance.
        """
        gp = _load_gpytorch()
        normalized_x = self._normalize_inputs(x)
        mean_x = self.mean_module(normalized_x)
        covar_x = self.covar_module(normalized_x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_distribution(self, inputs: torch.Tensor):
        """Evaluate the predictive distribution in no-gradient mode.

        Parameters
        ----------
        inputs : torch.Tensor
            Node-index column on the model device.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Likelihood-transformed predictive distribution.

        Notes
        -----
        The model and likelihood are placed in evaluation mode.
        """
        gp = _load_gpytorch()
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            return self.likelihood(self(inputs))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return posterior mean fitness for node-index inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Node-index column on the model device.

        Returns
        -------
        torch.Tensor
            Posterior mean with one value per input node on the model device.
        """
        posterior = self.predict_distribution(inputs)
        return posterior.mean


# Removed by issue #10 on integrated dev; retained only on this branch base.
class LandscapeNodeIndexInputAdapter(NodeIndexInputAdapter):  # noqa: D101
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
    """Fit an exact diffusion-prior GP on observed node fitness.

    This example is intended for small to medium landscapes where exact GP
    inference remains tractable. If ``train_indices`` is supplied, held-out
    finite targets remain available for evaluation but are not used for
    diffusion-scale fitting or GP likelihood optimization.

    Parameters
    ----------
    landscape : Any
        Landscapy fitness landscape with a fixed graph.
    target_layer : str
        Numeric fitness-layer name.
    aggregate_func : callable or None, default=numpy.mean
        Replicate aggregation function.
    train_indices : sequence of int, torch.Tensor, or None, optional
        Canonical nodes used for scale estimation and likelihood fitting.
    mask_tokens : sequence of str, default=DEFAULT_MASK_TOKENS
        Tokens treated as masked during scale estimation.
    drop_masked_sequences_for_t_map : bool, default=True
        Exclude masked training nodes when an estimable subset remains.
    t_min : float, default=0.01
        Minimum candidate diffusion scale.
    t_max : float, default=100.0
        Maximum candidate diffusion scale.
    epsilon : float, default=1e-8
        Numerical adjustment for diffusion estimation and covariance.
    method : str, default="grid"
        Landscapy diffusion-scale optimization method.
    grid_size : int, default=256
        Candidate count for grid optimization.
    prior : str, default="log_uniform"
        Diffusion-scale prior.
    bootstrap_samples : int, default=200
        Diffusion-scale bootstrap replicates.
    random_state : int or None, optional
        Diffusion-scale estimation seed.
    min_signal_variance : float, default=1e-6
        Lower bound for covariance signal variance.
    normalize_features : bool, default=False
        Normalize scalar node-index inputs.
    feature_normalization_eps : float, default=1e-8
        Minimum feature scale before replacement by one.
    training_iters : int, default=100
        Adam optimization iterations.
    learning_rate : float, default=0.1
        Adam learning rate.
    jitter : float, default=1e-4
        Kernel diagonal adjustment.
    mean_mode : {"constant", "zero"}, default="constant"
        GP mean-function family.
    device : str, torch.device, or None, optional
        Training device. Defaults to CPU.

    Returns
    -------
    DiffusionGPFitResult
        Fitted model, likelihood, full reproducibility artifacts, and loss
        history.

    Raises
    ------
    ImportError
        If GPyTorch or required Landscapy analysis functionality is missing.
    ValueError
        If landscape artifacts, model settings, or training inputs are invalid.

    Notes
    -----
    Exact GP time and memory scale cubically and quadratically with training
    nodes, respectively. More than 2,000 training observations triggers a
    resource warning. The graph, training subset, masking policy, diffusion
    settings, and random state determine the scientific prior and must be
    recorded with experimental results.
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
    """Predict diffusion-GP fitness for every landscape node.

    Parameters
    ----------
    landscape : Any
        Fixed landscape used to define the GP covariance.
    model : DiffusionPriorExactGP
        Fitted transductive GP model.
    layer_name : str, default="diffusion_gp_predicted_fitness"
        Requested numeric prediction-layer name.
    attach : bool, default=True
        Attach the prediction layer to a landscape.
    inplace : bool, default=True
        Attach to the supplied landscape when true or an independent copy when
        false. Ignored when ``attach`` is false.

    Returns
    -------
    BaseFitnessLayer or LandscapeInferenceResult
        Prediction layer when unattached or attached in place. For
        ``attach=True, inplace=False``, the copied landscape and attached layer.

    Raises
    ------
    ImportError
        If Landscapy numeric fitness support is unavailable.
    ValueError
        If model outputs, node count, or attachment metadata is invalid.
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
