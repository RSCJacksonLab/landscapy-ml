import numpy as np
import torch

from landscapyml.core.inference import (
    infer_fitness_layer_from_landscape,
    predict_landscape_records,
    predict_sequences,
    register_model_adapter,
    register_layer_adapter,
    register_model_layer_mapping,
)


class DummyLandscape:
    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
        self._attached = None
        self._active_embedding_domain = None
        self.embedding_model = None

    def get_embedding(self):
        return self._embeddings

    def attach(self, layer):
        self._attached = layer


class DummyProbModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cpu")

    def predict_with_uncertainty(self, x):
        mean = torch.full(
            (x.shape[0], self.num_classes),
            1.0 / self.num_classes,
            device=x.device,
        )
        var = torch.zeros_like(mean)
        return mean, var


def test_predict_sequences_uses_stubbed_embeddings(monkeypatch):
    # Stub embed_sequences to bypass heavy dependencies
    fake_embeddings = torch.randn(2, 3)
    monkeypatch.setattr(
        "landscapyml.core.inference.embed_sequences",
        lambda *args, **kwargs: (fake_embeddings, None, None),
    )
    model = DummyProbModel(num_classes=2)
    mean, var = predict_sequences(model, sequences=["AAA", "BBB"], model_name="ignored")
    assert mean.shape == (2, 2)
    assert var.shape == (2, 2)


def test_predict_landscape_records():
    model = DummyProbModel(num_classes=2)
    records = [
        {"embedding": torch.tensor([1.0, 0.0]), "fitness_tensors": {"label": 0}},
        {"embedding": torch.tensor([0.0, 1.0]), "fitness_tensors": {"label": 1}},
    ]
    mean, var = predict_landscape_records(model, records)
    assert mean.shape == (2, 2)
    assert var.shape == (2, 2)


def test_infer_fitness_layer_from_landscape_with_stub(monkeypatch):
    # Provide stub ProbabilisticCategoricalFitness for adapter to build
    class StubFitness:
        def __init__(self, name, probabilities, categories, metadata):
            self.name = name
            self.probabilities = probabilities
            self.categories = categories
            self.metadata = metadata

    monkeypatch.setattr(
        "landscapyml.core.adaptor.ProbabilisticCategoricalFitness", StubFitness
    )

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    landscape = DummyLandscape(embeddings)
    model = DummyProbModel(num_classes=2)
    register_model_layer_mapping(DummyProbModel, "prob_categorical", overwrite=True)

    layer = infer_fitness_layer_from_landscape(
        landscape,
        model,
        batch_size=1,
        attach=True,
        inplace=True,
        categories=["a", "b"],
    )
    assert isinstance(layer, StubFitness)
    assert hasattr(landscape, "_attached") and landscape._attached is layer


def test_custom_layer_adapter_registration(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, x):
            return x

    def adapter(out, categories, metadata, layer_name):
        return {"name": layer_name, "meta": metadata, "out": out}

    register_model_layer_mapping(DummyModel, "custom_layer", overwrite=True)
    register_layer_adapter("custom_layer", adapter, overwrite=True)

    dummy = DummyModel()

    # Reuse landscape helper for simplicity
    class MinimalLandscape(DummyLandscape):
        def attach(self, layer):
            self.layer = layer

    landscape = MinimalLandscape(np.array([[0.5]]))
    layer = infer_fitness_layer_from_landscape(
        landscape,
        dummy,
        batch_size=1,
        categories=["only"],
        attach=False,
    )
    assert layer["name"] == "predicted_fitness"


def test_model_adapter_registration(monkeypatch):
    class StubFitness:
        def __init__(self, name, probabilities, categories, metadata):
            self.name = name
            self.probabilities = probabilities
            self.categories = categories
            self.metadata = metadata

    monkeypatch.setattr(
        "landscapyml.core.adaptor.ProbabilisticCategoricalFitness", StubFitness
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    class DummyAdapter:
        layer_kind = "prob_categorical"

        def __init__(self, model):
            self.model = model

        def predict(self, inputs):
            mean = torch.ones(inputs.shape[0], 1)
            var = torch.zeros_like(mean)
            return {"mean": mean, "var": var}

    register_model_adapter(DummyModel, lambda model: DummyAdapter(model), overwrite=True)

    landscape = DummyLandscape(np.array([[0.5], [1.0]], dtype=float))
    layer = infer_fitness_layer_from_landscape(
        landscape,
        DummyModel(),
        batch_size=1,
        categories=["only"],
        attach=False,
    )
    assert isinstance(layer, StubFitness)
    assert layer.probabilities.shape == (2, 1)
