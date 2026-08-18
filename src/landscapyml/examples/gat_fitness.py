"""Graph-attention regression example for Landscapy fitness landscapes."""

from __future__ import annotations

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..core.adaptor import GraphTensorInputAdapter
from ..core.data import (
    LandscapeGraphRegressionDataModule,
    build_regression_graph_from_landscape,
)
from ..core.inference import infer_fitness_layer_from_landscape
from ..core.model_registry import register_model


def _load_gat_conv():
    try:
        from torch_geometric.nn import GATConv  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Graph attention examples require torch-geometric to be installed."
        ) from exc
    return GATConv


class GraphAttentionFitnessRegressor(pl.LightningModule):
    """Regress node fitness with graph attention convolutions.

    Parameters
    ----------
    in_channels : int or None, optional
        Explicit node-feature width. Takes priority over ``num_features``.
    num_features : int or None, optional
        Registry-compatible alias for the node-feature width.
    hidden_channels : int, default=64
        Hidden channels per attention head.
    num_layers : int, default=2
        Number of graph attention layers.
    heads : int, default=4
        Attention heads in hidden layers.
    dropout : float, default=0.1
        Attention and hidden-activation dropout probability.
    learning_rate : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=0.0
        Adam weight decay.

    Attributes
    ----------
    convs : torch.nn.ModuleList
        Ordered graph attention layers.
    feature_normalization_mean : torch.Tensor
        Optional per-feature mean loaded from the first input graph.
    feature_normalization_scale : torch.Tensor
        Optional positive per-feature scale loaded from the first input graph.
    layer_kind : str
        Output-adapter key ``"numeric"``.

    Raises
    ------
    ImportError
        If PyTorch Geometric is unavailable.
    ValueError
        If dimensions, layer count, or attention-head count are invalid.
    """

    layer_kind = "numeric"

    def __init__(
        self,
        *,
        in_channels: Optional[int] = None,
        num_features: Optional[int] = None,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = in_channels if in_channels is not None else num_features
        if input_dim is None or input_dim <= 0:
            raise ValueError(
                "GraphAttentionFitnessRegressor requires a positive input dimension."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if heads <= 0:
            raise ValueError("heads must be positive.")

        self.save_hyperparameters()
        GATConv = _load_gat_conv()
        self.dropout = float(dropout)
        self.convs = torch.nn.ModuleList()
        self.register_buffer("feature_normalization_mean", torch.empty(0))
        self.register_buffer("feature_normalization_scale", torch.empty(0))

        if num_layers == 1:
            self.convs.append(
                GATConv(input_dim, 1, heads=1, concat=False, dropout=self.dropout)
            )
        else:
            self.convs.append(
                GATConv(input_dim, hidden_channels, heads=heads, dropout=self.dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(
                        hidden_channels * heads,
                        hidden_channels,
                        heads=heads,
                        dropout=self.dropout,
                    )
                )
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    1,
                    heads=1,
                    concat=False,
                    dropout=self.dropout,
                )
            )

    def _maybe_load_feature_normalization(self, graph: Any) -> None:
        if self.feature_normalization_mean.numel() > 0:
            return
        mean = getattr(graph, "feature_normalization_mean", None)
        scale = getattr(graph, "feature_normalization_scale", None)
        if mean is None or scale is None:
            return
        mean_tensor = torch.as_tensor(
            mean, dtype=torch.float32, device=self.device
        ).view(-1)
        scale_tensor = torch.as_tensor(
            scale, dtype=torch.float32, device=self.device
        ).view(-1)
        if mean_tensor.numel() != scale_tensor.numel():
            raise ValueError(
                "Feature normalization mean and scale have different lengths."
            )
        if bool((scale_tensor <= 0).any()):
            raise ValueError("Feature normalization scale values must be positive.")
        self.feature_normalization_mean = mean_tensor.detach().clone()
        self.feature_normalization_scale = scale_tensor.detach().clone()

    def _normalize_features(self, graph: Any) -> torch.Tensor:
        self._maybe_load_feature_normalization(graph)
        x = graph.x
        if self.feature_normalization_mean.numel() == 0:
            return x
        mean = self.feature_normalization_mean.to(device=x.device, dtype=x.dtype)
        scale = self.feature_normalization_scale.to(device=x.device, dtype=x.dtype)
        if x.shape[-1] != mean.numel():
            raise ValueError(
                "Graph feature dimension does not match stored normalization stats."
            )
        return (x - mean) / scale

    def forward(self, graph: Any) -> torch.Tensor:
        """Predict one scalar for every graph node.

        Parameters
        ----------
        graph : Any
            PyTorch Geometric-style graph with floating ``x`` of shape
            ``(n_nodes, n_features)`` and integer ``edge_index`` of shape
            ``(2, n_edges)`` on the model device.

        Returns
        -------
        torch.Tensor
            Node predictions with shape ``(n_nodes,)`` on the input device.

        Raises
        ------
        ValueError
            If stored feature-normalization statistics are invalid or
            incompatible with graph feature width.
        """
        x = self._normalize_features(graph)
        edge_index = graph.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.view(-1)

    def predict(self, graph: Any) -> torch.Tensor:
        """Predict node fitness through :meth:`forward`.

        Parameters
        ----------
        graph : Any
            PyTorch Geometric-style graph on the model device.

        Returns
        -------
        torch.Tensor
            Node predictions with shape ``(n_nodes,)``.
        """
        return self(graph)

    def _step(self, graph: Any, mask_name: str, stage: str) -> Optional[torch.Tensor]:
        preds = self(graph).view(-1)
        target = graph.y.view(-1)
        mask = getattr(graph, mask_name, None)
        if mask is None:
            mask = torch.isfinite(target)
        else:
            mask = mask & torch.isfinite(target)
        if int(mask.sum().item()) == 0:
            return None

        loss = F.mse_loss(preds[mask], target[mask])
        mae = F.l1_loss(preds[mask], target[mask])
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{stage}/mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, graph: Any, batch_idx: int) -> torch.Tensor:
        """Compute and log masked training loss.

        Parameters
        ----------
        graph : Any
            Full graph carrying ``y`` and ``train_mask`` tensors.
        batch_idx : int
            Lightning batch index; unused for the single-graph dataset.

        Returns
        -------
        torch.Tensor
            Scalar mean-squared error on finite training targets.

        Raises
        ------
        RuntimeError
            If no finite supervised training node exists.
        """
        loss = self._step(graph, "train_mask", "train")
        if loss is None:
            raise RuntimeError(
                "Training graph does not contain any supervised train nodes."
            )
        return loss

    def validation_step(self, graph: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """Compute and log masked validation loss.

        Parameters
        ----------
        graph : Any
            Full graph carrying ``y`` and optional ``val_mask`` tensors.
        batch_idx : int
            Lightning batch index; unused for the single-graph dataset.

        Returns
        -------
        torch.Tensor or None
            Scalar mean-squared error, or ``None`` when no finite validation
            node is selected.
        """
        return self._step(graph, "val_mask", "val")

    def test_step(self, graph: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """Compute and log masked test loss.

        Parameters
        ----------
        graph : Any
            Full graph carrying ``y`` and optional ``test_mask`` tensors.
        batch_idx : int
            Lightning batch index; unused for the single-graph dataset.

        Returns
        -------
        torch.Tensor or None
            Scalar mean-squared error, or ``None`` when no finite test node is
            selected.
        """
        return self._step(graph, "test_mask", "test")

    def configure_optimizers(self):
        """Construct the Adam optimizer from saved hyperparameters.

        Returns
        -------
        torch.optim.Adam
            Optimizer over all trainable model parameters.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def attach_graph_attention_predictions(
    landscape: Any,
    model: GraphAttentionFitnessRegressor,
    *,
    layer_name: str = "gat_predicted_fitness",
    tokenizer: Any | str | None = None,
    attach: bool = True,
    inplace: bool = True,
) -> Any:
    """Predict graph-attention fitness for every landscape node.

    Parameters
    ----------
    landscape : Any
        Landscapy fitness landscape providing graph tensor export or a graph
        and sequences.
    model : GraphAttentionFitnessRegressor
        Trained graph attention model.
    layer_name : str, default="gat_predicted_fitness"
        Requested numeric prediction-layer name.
    tokenizer : Any, str, or None, optional
        Tokenizer forwarded to graph tensor export.
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
        If required Landscapy or PyTorch Geometric functionality is missing.
    ValueError
        If graph export, model output, or attachment metadata is invalid.
    """
    return infer_fitness_layer_from_landscape(
        landscape,
        model,
        batch_size=1,
        attach=attach,
        inplace=inplace,
        layer_name=layer_name,
        input_adapter=GraphTensorInputAdapter.name,
        input_adapter_kwargs={"tokenizer": tokenizer},
    )


register_model(
    "graph_attention_regressor",
    GraphAttentionFitnessRegressor,
    overwrite=True,
)
__all__ = [
    "GraphAttentionFitnessRegressor",
    "LandscapeGraphRegressionDataModule",
    "attach_graph_attention_predictions",
    "build_regression_graph_from_landscape",
]
