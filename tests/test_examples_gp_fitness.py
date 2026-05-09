import contextlib
import importlib
import sys
import types

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from landscapyml.core.adaptor import NodeIndexInputAdapter, resolve_input_adapter


class DummySequence:
    def __init__(self, seq):
        self._seq = np.asarray(list(seq), dtype=object)

    def to_array(self):
        return self._seq.copy()


class DummyNumericLayer:
    dtype = "numeric"

    def __init__(self, values):
        self._values = [list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] for v in values]
        self.metadata = {}

    def to_scalar(self, aggregate_func=np.mean):
        return np.asarray([aggregate_func(v) for v in self._values], dtype=float)

    def get_value(self, sequence_index):
        return self._values[sequence_index]


class DummyLandscape:
    def __init__(self):
        self.graph = nx.path_graph(4)
        for node, seq in zip(self.graph.nodes(), ["AA", "AX", "AC", "AD"]):
            self.graph.nodes[node]["sequence"] = DummySequence(seq)
        self.sequences = [self.graph.nodes[node]["sequence"] for node in self.graph.nodes()]
        self.fitness_layers = {
            "score": DummyNumericLayer([1.0, 2.0, 3.0, np.nan]),
        }
        self._node_order = list(self.graph.nodes())
        self.embeddings = {}
        self._emb_arr_key = "emb_arr"
        self._active_embedding_domain = None
        self._embedding_metadata = {}


class FakeNumericFitness:
    def __init__(self, name, values, metadata=None):
        self.name = name
        self.values = values
        self.metadata = metadata or {}

    @property
    def dtype(self):
        return "numeric"

    def to_scalar(self, aggregate_func=np.mean):
        return np.asarray([aggregate_func(v) for v in self.values], dtype=float)

    def get_value(self, sequence_index):
        return self.values[sequence_index]


class FakeFitnessLandscape:
    def __init__(
        self,
        *,
        sequences,
        graph,
        fitness_layers,
        embeddings=None,
        emb_arr_key="emb_arr",
        active_embedding_domain=None,
        embedding_metadata=None,
    ):
        self.sequences = sequences
        self.graph = graph
        self.fitness_layers = fitness_layers
        self.embeddings = embeddings or {}
        self._emb_arr_key = emb_arr_key
        self._active_embedding_domain = active_embedding_domain
        self._embedding_metadata = embedding_metadata or {}
        self._node_order = list(graph.nodes())

    def view(self, name):
        self.active_layer_name = name
        return self.fitness_layers[name]


def test_build_diffusion_gp_artifacts_drops_masked_nodes_for_t_map(monkeypatch):
    from landscapyml.examples import gp_fitness

    called = {}

    monkeypatch.setattr(gp_fitness, "NumericFitness", FakeNumericFitness)
    monkeypatch.setattr(gp_fitness, "FitnessLandscape", FakeFitnessLandscape)

    def fake_fit(sub_landscape, **kwargs):  # noqa: ARG001
        called["n_sequences"] = len(sub_landscape.sequences)
        return {"t_map": 2.5}

    def fake_eigendecomp(graph, matrix="norm_laplacian"):  # noqa: ARG001
        n = graph.number_of_nodes()
        return np.linspace(0.0, 1.0, n), np.eye(n)

    monkeypatch.setattr(gp_fitness, "compute_ruggedness_diffusion_scale", fake_fit)
    monkeypatch.setattr(gp_fitness, "eigenmode_decomposition", fake_eigendecomp)

    artifacts = gp_fitness.build_diffusion_gp_artifacts_from_landscape(
        DummyLandscape(),
        target_layer="score",
        mask_tokens=("X",),
        normalize_features=True,
    )

    assert called["n_sequences"] == 2
    assert artifacts.t_map == 2.5
    assert artifacts.fit_indices.tolist() == [0, 2]
    assert artifacts.train_indices.tolist() == [0, 1, 2]
    assert artifacts.predict_mask.tolist() == [False, False, False, True]
    assert artifacts.covariance_matrix.shape == (4, 4)
    assert artifacts.normalize_features is True
    assert artifacts.feature_normalization_mean == 1.0
    assert np.isclose(
        artifacts.feature_normalization_scale,
        np.std(np.asarray([0.0, 1.0, 2.0]), ddof=0),
    )


def _install_fake_gpytorch(monkeypatch):
    class FakeKernel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeExactGP(torch.nn.Module):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__()
            self.train_inputs = (train_x,)
            self.train_targets = train_y
            self.likelihood = likelihood

    class FakeGaussianLikelihood(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, mvn):
            return mvn

    class FakeConstantMean(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return self.bias.expand(x.shape[0])

    class FakeZeroMean(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    class FakeMultivariateNormal:
        def __init__(self, mean, covariance_matrix):
            self.mean = mean
            self.covariance_matrix = covariance_matrix

    class FakeExactMarginalLogLikelihood:
        def __init__(self, likelihood, model):  # noqa: ARG002
            pass

        def __call__(self, output, target):
            return -F.mse_loss(output.mean, target)

    fake_gpytorch = types.SimpleNamespace(
        kernels=types.SimpleNamespace(Kernel=FakeKernel),
        models=types.SimpleNamespace(ExactGP=FakeExactGP),
        likelihoods=types.SimpleNamespace(GaussianLikelihood=FakeGaussianLikelihood),
        means=types.SimpleNamespace(
            ConstantMean=FakeConstantMean,
            ZeroMean=FakeZeroMean,
        ),
        distributions=types.SimpleNamespace(
            MultivariateNormal=FakeMultivariateNormal,
        ),
        mlls=types.SimpleNamespace(
            ExactMarginalLogLikelihood=FakeExactMarginalLogLikelihood,
        ),
        settings=types.SimpleNamespace(
            fast_pred_var=lambda: contextlib.nullcontext(),
        ),
    )
    monkeypatch.setitem(sys.modules, "gpytorch", fake_gpytorch)


def test_diffusion_prior_gp_predict_and_fit(monkeypatch):
    _install_fake_gpytorch(monkeypatch)
    module = importlib.import_module("landscapyml.examples.gp_fitness")
    module = importlib.reload(module)

    covariance = torch.eye(3)
    model = module.DiffusionPriorExactGP(
        train_x=torch.tensor([[0.0], [2.0]]),
        train_y=torch.tensor([1.0, 2.0]),
        covariance_matrix=covariance,
        normalize_features=True,
        feature_normalization_mean=1.0,
        feature_normalization_scale=1.0,
    )
    preds = model.predict(torch.tensor([[0.0], [1.0], [2.0]]))
    assert preds.shape == (3,)
    assert torch.equal(
        model.covar_module._to_index_tensor(torch.tensor([[-1.0], [0.0], [1.0]])),
        torch.tensor([0, 1, 2]),
    )

    fake_artifacts = module.DiffusionGPArtifacts(
        covariance_matrix=covariance,
        all_inputs=torch.tensor([[0.0], [1.0], [2.0]]),
        train_inputs=torch.tensor([[0.0], [2.0]]),
        train_targets=torch.tensor([1.0, 2.0]),
        full_targets=torch.tensor([1.0, float("nan"), 2.0]),
        observed_mask=torch.tensor([True, False, True]),
        predict_mask=torch.tensor([False, True, False]),
        fit_indices=torch.tensor([0, 2]),
        train_indices=torch.tensor([0, 2]),
        t_map=1.5,
        signal_variance=0.5,
        mask_tokens=("X",),
        normalize_features=True,
        feature_normalization_mean=1.0,
        feature_normalization_scale=1.0,
    )

    monkeypatch.setattr(
        module,
        "build_diffusion_gp_artifacts_from_landscape",
        lambda *args, **kwargs: fake_artifacts,
    )
    result = module.fit_diffusion_prior_gp(
        DummyLandscape(),
        target_layer="score",
        training_iters=2,
        learning_rate=0.05,
    )
    assert len(result.losses) == 2
    assert result.artifacts.t_map == 1.5
    assert result.model.normalize_features is True


def test_gp_example_registers_core_node_index_adapter_alias(monkeypatch):
    _install_fake_gpytorch(monkeypatch)
    module = importlib.import_module("landscapyml.examples.gp_fitness")
    importlib.reload(module)

    adapter = resolve_input_adapter("landscape_node_index")
    assert isinstance(adapter, NodeIndexInputAdapter)
