from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..core.adaptor import GraphTensorInputAdapter, register_input_adapter
from ..core.inference import infer_fitness_layer_from_landscape
from ..core.trainer import register_data, register_model


def _load_gat_conv():
    try:
        from torch_geometric.nn import GATConv  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Graph attention examples require torch-geometric to be installed."
        ) from exc
    return GATConv


def _aggregate_numeric_targets(
    layer: Any, *, aggregate_func: Optional[Callable[..., Any]] = np.mean
) -> np.ndarray:
    if getattr(layer, "dtype", None) != "numeric":
        raise ValueError(
            "Graph attention example currently supports numeric fitness layers only."
        )
    if aggregate_func is None:
        values = layer.to_scalar()
    else:
        try:
            values = layer.to_scalar(aggregate_func=aggregate_func)
        except TypeError:
            values = layer.to_scalar()
    return np.asarray(values, dtype=float).reshape(-1)


def _build_mask(
    num_nodes: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if indices.numel() > 0:
        mask[indices] = True
    return mask


def _collate_single_graph(items: list[Any]) -> Any:
    return items[0]


def build_regression_graph_from_landscape(
    landscape: Any,
    *,
    target_layer: str,
    tokenizer: Any | str | None = None,
    aggregate_func: Optional[Callable[..., Any]] = np.mean,
    val_fraction: float = 0.0,
    test_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> Any:
    """
    Build a PyG-style single-graph regression object from a ``FitnessLandscape``.

    Known numeric fitness values are used as supervision targets; unknown values
    should be encoded as ``NaN`` in the source layer and will be excluded from
    the training masks automatically.
    """

    if not hasattr(landscape, "to_graph_tensor"):
        raise ValueError("Landscape must implement to_graph_tensor().")
    layers = getattr(landscape, "fitness_layers", None)
    if not isinstance(layers, Mapping) or target_layer not in layers:
        raise ValueError(f"Landscape does not contain target layer '{target_layer}'.")

    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    graph = landscape.to_graph_tensor(tokenizer=tokenizer)
    values = _aggregate_numeric_targets(
        layers[target_layer],
        aggregate_func=aggregate_func,
    )
    y = torch.as_tensor(values, dtype=torch.float32).view(-1)
    if y.shape[0] != int(getattr(graph, "num_nodes", y.shape[0])):
        raise ValueError("Target layer length does not match graph node count.")

    known_idx = torch.nonzero(torch.isfinite(y), as_tuple=False).view(-1)
    if known_idx.numel() == 0:
        raise ValueError("No known numeric fitness values were found for training.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(known_idx.numel(), generator=generator)
    known_idx = known_idx[perm]

    n_known = known_idx.numel()
    n_test = int(round(float(test_fraction) * n_known))
    n_val = int(round(float(val_fraction) * n_known))
    n_train = n_known - n_val - n_test
    if n_train <= 0:
        raise ValueError("At least one known node must remain in the training mask.")

    train_idx = known_idx[:n_train]
    val_idx = known_idx[n_train : n_train + n_val]
    test_idx = known_idx[n_train + n_val :]

    num_nodes = y.shape[0]
    graph.y = y
    graph.target_layer = target_layer
    graph.known_mask = _build_mask(num_nodes, known_idx)
    graph.train_mask = _build_mask(num_nodes, train_idx)
    graph.val_mask = _build_mask(num_nodes, val_idx)
    graph.test_mask = _build_mask(num_nodes, test_idx)
    graph.predict_mask = ~graph.known_mask
    return graph


class _SingleGraphDataset(Dataset):
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        if idx != 0:
            raise IndexError(idx)
        return self.graph


class LandscapeGraphRegressionDataModule(pl.LightningDataModule):
    """
    Minimal DataModule for node regression on a single landscape graph.
    """

    def __init__(
        self,
        *,
        train_graph: Any,
        val_graph: Any = None,
        test_graph: Any = None,
        predict_graph: Any = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.test_graph = test_graph
        self.predict_graph = predict_graph
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @classmethod
    def from_landscape(
        cls,
        *,
        landscape: Any,
        target_layer: str,
        tokenizer: Any | str | None = None,
        aggregate_func: Optional[Callable[..., Any]] = np.mean,
        val_fraction: float = 0.0,
        test_fraction: float = 0.0,
        seed: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> "LandscapeGraphRegressionDataModule":
        graph = build_regression_graph_from_landscape(
            landscape,
            target_layer=target_layer,
            tokenizer=tokenizer,
            aggregate_func=aggregate_func,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        return cls(
            train_graph=graph,
            val_graph=graph if getattr(graph, "val_mask", None) is not None else None,
            test_graph=graph if getattr(graph, "test_mask", None) is not None else None,
            predict_graph=graph,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _loader(self, graph: Any) -> Optional[DataLoader]:
        if graph is None:
            return None
        dataset = _SingleGraphDataset(graph)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate_single_graph,
        )

    def train_dataloader(self) -> DataLoader:
        loader = self._loader(self.train_graph)
        if loader is None:
            raise RuntimeError("Training graph was not initialized.")
        return loader

    def val_dataloader(self) -> Optional[DataLoader]:
        graph = self.val_graph
        if (
            graph is not None
            and hasattr(graph, "val_mask")
            and int(graph.val_mask.sum().item()) == 0
        ):
            graph = None
        return self._loader(graph)

    def test_dataloader(self) -> Optional[DataLoader]:
        graph = self.test_graph
        if (
            graph is not None
            and hasattr(graph, "test_mask")
            and int(graph.test_mask.sum().item()) == 0
        ):
            graph = None
        return self._loader(graph)

    def predict_dataloader(self) -> Optional[DataLoader]:
        return self._loader(self.predict_graph)


class GraphAttentionFitnessRegressor(pl.LightningModule):
    """
    Example graph attention regressor for semi-supervised node fitness prediction.
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
            raise ValueError("GraphAttentionFitnessRegressor requires a positive input dimension.")
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

    def forward(self, graph: Any) -> torch.Tensor:
        x = graph.x
        edge_index = graph.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.view(-1)

    def predict(self, graph: Any) -> torch.Tensor:
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
        loss = self._step(graph, "train_mask", "train")
        if loss is None:
            raise RuntimeError(
                "Training graph does not contain any supervised train nodes."
            )
        return loss

    def validation_step(self, graph: Any, batch_idx: int) -> Optional[torch.Tensor]:
        return self._step(graph, "val_mask", "val")

    def test_step(self, graph: Any, batch_idx: int) -> Optional[torch.Tensor]:
        return self._step(graph, "test_mask", "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


class LandscapeGraphInputAdapter(GraphTensorInputAdapter):
    name = "landscape_graph"


def attach_graph_attention_predictions(
    landscape: Any,
    model: GraphAttentionFitnessRegressor,
    *,
    layer_name: str = "gat_predicted_fitness",
    tokenizer: Any | str | None = None,
    attach: bool = True,
    inplace: bool = True,
) -> Any:
    """
    Run graph-model inference on a landscape and attach the predicted numeric layer.
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


register_input_adapter(
    LandscapeGraphInputAdapter.name,
    LandscapeGraphInputAdapter,
    overwrite=True,
)
register_model(
    "graph_attention_regressor",
    GraphAttentionFitnessRegressor,
    overwrite=True,
)
register_data(
    "landscape_graph_regression",
    LandscapeGraphRegressionDataModule.from_landscape,
    overwrite=True,
)


__all__ = [
    "GraphAttentionFitnessRegressor",
    "LandscapeGraphInputAdapter",
    "LandscapeGraphRegressionDataModule",
    "attach_graph_attention_predictions",
    "build_regression_graph_from_landscape",
]
