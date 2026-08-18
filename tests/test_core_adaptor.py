import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from landscapyml.core.adaptor import (
    DefaultModelAdapter,
    EmbeddingInputAdapter,
    FunctionOutputAdapter,
    GraphTensorInputAdapter,
    NodeIndexInputAdapter,
    NumericOutputAdapter,
    ProbCategoricalOutputAdapter,
    infer_device,
    normalize_adapter_outputs,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    resolve_embedding_info,
    resolve_input_adapter,
    resolve_model_adapter,
    resolve_output_adapter,
)


class DummyLandscape:
    def __init__(self, embeddings: np.ndarray, domain: str | None = None, model: str | None = None):
        self._embeddings = embeddings
        self._active_embedding_domain = domain
        self.embedding_model = model

    def get_embedding(self):
        return self._embeddings


def test_normalize_adapter_outputs_variants_and_errors():
    mapping = {"mean": torch.tensor([1.0])}
    assert normalize_adapter_outputs(mapping, "prob_categorical") == mapping

    tensor_out = torch.tensor([2.0])
    out = normalize_adapter_outputs(tensor_out, "numeric")
    assert out["output"].shape == (1,)

    tuple_out = (torch.ones(1, 2), torch.zeros(1, 2))
    out = normalize_adapter_outputs(tuple_out, "prob_categorical")
    assert set(out) == {"mean", "var"}

    with pytest.raises(ValueError):
        normalize_adapter_outputs([1, 2, 3], "numeric")


def test_default_model_adapter_handles_probabilistic_output():
    class DummyModel(torch.nn.Module):
        def predict(self, x):
            return torch.ones(x.shape[0], 1), torch.zeros(x.shape[0], 1)

    register_model_layer_mapping(DummyModel, "prob_categorical", overwrite=True)
    adapter = resolve_model_adapter(DummyModel())
    outputs = adapter.predict(torch.zeros(2, 3))
    assert set(outputs) == {"mean", "var"}
    assert outputs["mean"].shape[0] == 2


def test_resolve_model_adapter_prefers_registered_factory():
    class CustomModel(torch.nn.Module):
        def forward(self, x):
            return x

    class CustomAdapter:
        layer_kind = "custom"

        def __init__(self, model):
            self.model = model

        def predict(self, x):
            return {"output": x + 1}

    register_model_adapter(CustomModel, lambda model: CustomAdapter(model), overwrite=True)
    adapter = resolve_model_adapter(CustomModel())
    assert isinstance(adapter, CustomAdapter)


def test_resolve_model_adapter_raises_for_unmapped_class():
    class UnmappedModel(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="UnmappedModel"):
        resolve_model_adapter(UnmappedModel())


def test_resolve_model_adapter_inherits_base_layer_mapping():
    class BaseModel(torch.nn.Module):
        pass

    class DerivedModel(BaseModel):
        pass

    register_model_layer_mapping(BaseModel, "base_kind", overwrite=True)

    assert resolve_model_adapter(BaseModel()).layer_kind == "base_kind"
    assert resolve_model_adapter(DerivedModel()).layer_kind == "base_kind"


def test_resolve_model_adapter_prefers_subclass_layer_mapping():
    class BaseModel(torch.nn.Module):
        pass

    class DerivedModel(BaseModel):
        pass

    register_model_layer_mapping(BaseModel, "base_kind", overwrite=True)
    register_model_layer_mapping(DerivedModel, "derived_kind", overwrite=True)

    assert resolve_model_adapter(DerivedModel()).layer_kind == "derived_kind"


def test_resolve_model_adapter_uses_python_mro_for_multiple_inheritance():
    class LeftModel(torch.nn.Module):
        pass

    class RightModel(torch.nn.Module):
        pass

    class LeftFirstModel(LeftModel, RightModel):
        pass

    class RightFirstModel(RightModel, LeftModel):
        pass

    register_model_layer_mapping(LeftModel, "left_kind", overwrite=True)
    register_model_layer_mapping(RightModel, "right_kind", overwrite=True)

    assert resolve_model_adapter(LeftFirstModel()).layer_kind == "left_kind"
    assert resolve_model_adapter(RightFirstModel()).layer_kind == "right_kind"


def test_embedding_input_adapter_yields_batches_and_metadata():
    emb = np.random.randn(5, 4).astype(np.float32)
    land = DummyLandscape(embeddings=emb, domain="sequence", model="esm-test")
    adapter = EmbeddingInputAdapter()
    batches = list(adapter.iter_batches(land, batch_size=2))
    assert len(batches) == 3  # 2 + 2 + 1
    first = adapter.to_model_inputs(batches[0])
    assert torch.is_tensor(first)
    meta = adapter.metadata(land)
    assert meta["input_adapter"] == "embedding"
    assert meta["embedding_model"] == "esm-test"
    assert meta["embedding_domain"] == "sequence"


def test_resolve_embedding_info_propagates_metadata_errors():
    class BrokenMetadataLandscape:
        active_embedding_domain = "sequence"

        def get_embedding_metadata(self, domain):  # noqa: ARG002
            raise RuntimeError("metadata store is corrupt")

    with pytest.raises(RuntimeError, match="metadata store is corrupt"):
        resolve_embedding_info(BrokenMetadataLandscape())


def test_graph_tensor_input_adapter_uses_landscape_graph_export():
    class GraphLike:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class GraphLandscape:
        def __init__(self):
            self.calls = []

        def to_graph_tensor(self, tokenizer=None):
            self.calls.append(tokenizer)
            return GraphLike()

    landscape = GraphLandscape()
    adapter = GraphTensorInputAdapter(tokenizer="esm-test")
    batches = list(adapter.iter_batches(landscape, batch_size=1))
    assert len(batches) == 1
    graph = adapter.to_model_inputs(batches[0], device=torch.device("cpu"))
    assert landscape.calls == ["esm-test"]
    assert graph.moved_to == torch.device("cpu")
    assert adapter.metadata(landscape)["graph_tensor"] is True


def test_node_index_input_adapter_batches_landscape_indices():
    landscape = SimpleNamespace(sequences=["A", "B", "C"])
    adapter = NodeIndexInputAdapter()
    batches = list(adapter.iter_batches(landscape, batch_size=2))
    assert len(batches) == 2
    first = adapter.to_model_inputs(batches[0])
    second = adapter.to_model_inputs(batches[1])
    assert first.shape == (2, 1)
    assert second.shape == (1, 1)
    assert first[:, 0].tolist() == [0.0, 1.0]
    assert second[:, 0].tolist() == [2.0]


def test_input_adapter_registry_unknown_name_errors():
    with pytest.raises(ValueError):
        resolve_input_adapter("nonexistent")

    class DummyAdapter(EmbeddingInputAdapter):
        name = "dummy_adapter"

    register_input_adapter(DummyAdapter.name, lambda: DummyAdapter(), overwrite=True)
    resolved = resolve_input_adapter("dummy_adapter")
    assert isinstance(resolved, DummyAdapter)


def test_prob_categorical_output_adapter_builds_layer(monkeypatch):
    class StubFitness:
        def __init__(self, name, probabilities, categories, metadata):
            self.name = name
            self.probabilities = probabilities
            self.categories = categories
            self.metadata = metadata

    monkeypatch.setattr(
        "landscapyml.core.adaptor.ProbabilisticCategoricalFitness", StubFitness
    )
    adapter = ProbCategoricalOutputAdapter()
    outputs = {
        "mean": torch.tensor([[0.7, 0.3]]),
        "var": torch.tensor([[0.05, 0.02]]),
    }
    layer = adapter.to_layer(outputs, ["yes", "no"], {"source": "test"}, "layer_name")
    assert isinstance(layer, StubFitness)
    assert layer.categories == ["yes", "no"]
    assert "variance" in layer.metadata


def test_prob_categorical_output_adapter_detaches_tensors(monkeypatch):
    class StubFitness:
        def __init__(self, name, probabilities, categories, metadata):
            self.name = name
            self.probabilities = probabilities
            self.categories = categories
            self.metadata = metadata

    monkeypatch.setattr(
        "landscapyml.core.adaptor.ProbabilisticCategoricalFitness", StubFitness
    )
    mean = torch.tensor([[0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    var = torch.tensor([[0.05, 0.02]], dtype=torch.float64, requires_grad=True)

    layer = ProbCategoricalOutputAdapter().to_layer(
        {"mean": mean, "var": var},
        ["yes", "no"],
        {},
        "layer_name",
    )

    assert layer.probabilities.shape == mean.shape
    assert layer.probabilities.dtype == np.float64
    assert layer.metadata["variance"].shape == var.shape
    assert layer.metadata["variance"].dtype == np.float64


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="No supported accelerator is available.",
)
def test_prob_categorical_output_adapter_moves_accelerator_tensors_to_cpu(
    monkeypatch,
):
    class StubFitness:
        def __init__(self, name, probabilities, categories, metadata):
            self.name = name
            self.probabilities = probabilities
            self.categories = categories
            self.metadata = metadata

    monkeypatch.setattr(
        "landscapyml.core.adaptor.ProbabilisticCategoricalFitness", StubFitness
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    mean = torch.tensor([[0.7, 0.3]], device=device, requires_grad=True)
    var = torch.tensor([[0.05, 0.02]], device=device, requires_grad=True)

    layer = ProbCategoricalOutputAdapter().to_layer(
        {"mean": mean, "var": var},
        ["yes", "no"],
        {},
        "layer_name",
    )

    np.testing.assert_allclose(layer.probabilities, [[0.7, 0.3]])
    np.testing.assert_allclose(layer.metadata["variance"], [[0.05, 0.02]])


@pytest.mark.parametrize(
    ("outputs", "error_type", "message"),
    [
        ({"mean": [[0.7, 0.3]]}, TypeError, "non-tensor 'mean'"),
        ({"mean": torch.tensor([0.7, 0.3])}, ValueError, "expects 'mean'"),
        (
            {"mean": torch.tensor([[0.7, 0.3]]), "var": [[0.05, 0.02]]},
            TypeError,
            "non-tensor 'var'",
        ),
        (
            {
                "mean": torch.tensor([[0.7, 0.3]]),
                "var": torch.tensor([[0.05], [0.02]]),
            },
            ValueError,
            "same shape",
        ),
    ],
)
def test_prob_categorical_output_adapter_validates_tensor_shapes(
    outputs, error_type, message
):
    with pytest.raises(error_type, match=message):
        ProbCategoricalOutputAdapter().to_layer(
            outputs,
            ["yes", "no"],
            {},
            "layer_name",
        )


def test_output_adapter_registry_with_function_wrapper():
    register_layer_adapter(
        "custom_kind",
        lambda outputs, categories, metadata, layer_name: {
            "name": layer_name,
            "meta": metadata,
            "out": outputs,
        },
        overwrite=True,
    )
    out_adapter = resolve_output_adapter("custom_kind")
    layer = out_adapter.to_layer({"value": 1}, None, {"x": 1}, "my_layer")
    assert layer["name"] == "my_layer"
    assert layer["meta"]["x"] == 1


def test_numeric_output_adapter_builds_layer(monkeypatch):
    class StubFitness:
        @classmethod
        def from_tensor(cls, name, tensor, metadata):
            inst = cls()
            inst.name = name
            inst.tensor = tensor
            inst.metadata = metadata
            return inst

    monkeypatch.setattr("landscapyml.core.adaptor.NumericFitness", StubFitness)
    adapter = NumericOutputAdapter()
    layer = adapter.to_layer(
        {"output": torch.tensor([0.1, 0.2])},
        None,
        {"source": "test"},
        "fitness_pred",
    )
    assert isinstance(layer, StubFitness)
    assert layer.tensor.shape == (2, 1)
    assert layer.metadata["source"] == "test"


def test_infer_device_prefers_explicit_device_attribute():
    obj = SimpleNamespace(device="cuda:5")
    dev = infer_device(obj)
    assert dev == torch.device("cuda:5")

    class ParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

    model = ParamModel()
    dev2 = infer_device(model)
    assert dev2 == model.linear.weight.device
