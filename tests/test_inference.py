import numpy as np
import pytest
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


class PlainProbModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.inference_mode_enabled = False

    def predict_with_uncertainty(self, inputs):
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        mean = torch.softmax(self.linear(inputs), dim=-1)
        return mean, torch.zeros_like(mean)


def test_predict_sequences_supports_plain_module_and_disables_grad(monkeypatch):
    fake_embeddings = torch.randn(2, 2)
    monkeypatch.setattr(
        "landscapyml.core.inference.embed_sequences",
        lambda *args, **kwargs: (fake_embeddings, None, None),
    )
    model = PlainProbModel()

    mean, var = predict_sequences(model, sequences=["AAA", "BBB"])

    assert not model.training
    assert model.inference_mode_enabled
    assert mean.device.type == "cpu"
    assert var.device.type == "cpu"
    assert not mean.requires_grad
    assert not var.requires_grad


def test_predict_landscape_records_accepts_generator_for_plain_module():
    model = PlainProbModel()
    records = (
        {"embedding": torch.tensor(feature)}
        for feature in ([1.0, 0.0], [0.0, 1.0])
    )

    mean, var = predict_landscape_records(model, records)

    assert mean.shape == (2, 2)
    assert var.shape == (2, 2)
    assert model.inference_mode_enabled


def test_prediction_helpers_detach_preexisting_grad_enabled_outputs(monkeypatch):
    class GradOutputModel:
        def __init__(self):
            self.mean = torch.tensor([[0.6, 0.4]], requires_grad=True)
            self.var = torch.tensor([[0.1, 0.2]], requires_grad=True)

        def eval(self):
            return self

        def predict_with_uncertainty(self, inputs):
            return self.mean, self.var

    monkeypatch.setattr(
        "landscapyml.core.inference.embed_sequences",
        lambda *args, **kwargs: (torch.randn(1, 2), None, None),
    )

    mean, var = predict_sequences(GradOutputModel(), sequences=["AAA"])

    assert not mean.requires_grad
    assert not var.requires_grad


def test_prediction_helpers_reject_empty_inputs(monkeypatch):
    def unexpected_embedding(*args, **kwargs):
        raise AssertionError("empty inputs should fail before embedding")

    monkeypatch.setattr(
        "landscapyml.core.inference.embed_sequences", unexpected_embedding
    )
    model = PlainProbModel()

    with pytest.raises(ValueError, match="sequences.*at least one"):
        predict_sequences(model, sequences=[])
    with pytest.raises(ValueError, match="records.*at least one"):
        predict_landscape_records(model, iter(()))


def test_predict_landscape_records_rejects_inconsistent_feature_shapes():
    records = (
        {"embedding": torch.tensor(feature)}
        for feature in ([1.0, 0.0], [0.0, 1.0, 2.0])
    )

    with pytest.raises(ValueError, match="Record 1 feature shape"):
        predict_landscape_records(PlainProbModel(), records)


@pytest.mark.parametrize(
    ("outputs", "error_type", "message"),
    [
        ((torch.ones(1, 2),), ValueError, "exactly two outputs"),
        (([[0.5, 0.5]], torch.zeros(1, 2)), TypeError, "torch.Tensor"),
        (
            (torch.ones(2, 2), torch.zeros(2, 2)),
            ValueError,
            "Mean prediction batch dimension",
        ),
        (
            (torch.ones(1, 2), torch.zeros(2, 2)),
            ValueError,
            "Variance prediction batch dimension",
        ),
        (
            (torch.ones(1, 2), torch.zeros(1, 1)),
            ValueError,
            "same shape",
        ),
    ],
)
def test_predict_landscape_records_validates_uncertainty_outputs(
    outputs, error_type, message
):
    class InvalidOutputModel:
        def eval(self):
            return self

        def predict_with_uncertainty(self, inputs):
            return outputs

    records = [{"embedding": torch.tensor([1.0, 0.0])}]

    with pytest.raises(error_type, match=message):
        predict_landscape_records(InvalidOutputModel(), records)


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
