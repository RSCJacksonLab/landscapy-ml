import sys
import types

import numpy as np
import pytest
import torch

from landscapyml.core.adaptor import GraphTensorInputAdapter, resolve_input_adapter
from landscapyml.core.data import build_regression_graph_from_landscape


class FakeGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__()
        effective_out = out_channels * heads if concat else out_channels
        self.linear = torch.nn.Linear(in_channels, effective_out)

    def forward(self, x, edge_index):  # noqa: ARG002 - fake test double ignores graph edges
        return self.linear(x)


class DummyNumericLayer:
    dtype = "numeric"

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def to_scalar(self, aggregate_func=np.mean):  # noqa: ARG002 - aggregate unused in stub
        return self._values


class DummyGraph:
    def __init__(self):
        self.x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        self.num_nodes = 3

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        for name in (
            "y",
            "known_mask",
            "train_mask",
            "val_mask",
            "test_mask",
            "predict_mask",
            "feature_normalization_mean",
            "feature_normalization_scale",
        ):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self


class DummyLandscape:
    def __init__(self):
        self.fitness_layers = {"score": DummyNumericLayer([0.5, np.nan, 1.5])}

    def to_graph_tensor(self, tokenizer=None):  # noqa: ARG002 - tokenizer unused in stub
        return DummyGraph()


def test_build_regression_graph_from_landscape(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch_geometric.nn",
        types.SimpleNamespace(GATConv=FakeGATConv),
    )

    graph = build_regression_graph_from_landscape(
        DummyLandscape(),
        target_layer="score",
        val_fraction=0.0,
        test_fraction=0.0,
        seed=0,
    )
    assert graph.y.shape == (3,)
    assert graph.known_mask.tolist() == [True, False, True]
    assert graph.predict_mask.tolist() == [False, True, False]
    assert graph.train_mask.tolist() == [True, False, True]


def test_build_regression_graph_adds_training_feature_normalization(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch_geometric.nn",
        types.SimpleNamespace(GATConv=FakeGATConv),
    )

    from landscapyml.models.gat_fitness import (
        GraphAttentionFitnessRegressor,
    )

    graph = build_regression_graph_from_landscape(
        DummyLandscape(),
        target_layer="score",
        normalize_features=True,
        seed=0,
    )

    assert graph.normalize_features is True
    assert torch.allclose(graph.feature_normalization_mean, torch.tensor([1.0, 0.5]))
    assert torch.allclose(graph.feature_normalization_scale, torch.tensor([1.0, 0.5]))

    model = GraphAttentionFitnessRegressor(
        num_features=2, hidden_channels=4, num_layers=1
    )
    normalized = model._normalize_features(graph)
    assert torch.allclose(
        normalized,
        torch.tensor(
            [
                [0.0, -1.0],
                [-1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )


def test_graph_attention_regressor_forward(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch_geometric.nn",
        types.SimpleNamespace(GATConv=FakeGATConv),
    )

    from landscapyml.models.gat_fitness import GraphAttentionFitnessRegressor

    graph = DummyGraph()
    model = GraphAttentionFitnessRegressor(
        num_features=2, hidden_channels=4, num_layers=2
    )
    out = model(graph)
    assert out.shape == (3,)


def test_graph_example_uses_only_core_graph_adapter(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch_geometric.nn",
        types.SimpleNamespace(GATConv=FakeGATConv),
    )

    __import__("landscapyml.models.gat_fitness")
    adapter = resolve_input_adapter("graph_tensor")
    assert isinstance(adapter, GraphTensorInputAdapter)
    with pytest.raises(ValueError, match="Unknown input adapter 'landscape_graph'"):
        resolve_input_adapter("landscape_graph")
