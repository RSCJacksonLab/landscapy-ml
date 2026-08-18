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
    export_landscape_records,
    infer_device,
    normalize_adapter_outputs,
    register_input_adapter,
    register_layer_adapter,
    register_model_adapter,
    register_model_layer_mapping,
    resolve_input_adapter,
    resolve_model_adapter,
    resolve_output_adapter,
)


class DummyLandscape:
    def __init__(
        self,
        embeddings: np.ndarray,
        domain: str | None = None,
        model: str | None = None,
    ):
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

    register_model_adapter(
        CustomModel, lambda model: CustomAdapter(model), overwrite=True
    )
    adapter = resolve_model_adapter(CustomModel())
    assert isinstance(adapter, CustomAdapter)


def test_resolve_model_adapter_raises_for_unmapped_class():
    class UnmappedModel(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError):
        resolve_model_adapter(UnmappedModel())


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


def test_export_landscape_records_selects_renames_and_maps_categories():
    class CategoryLayer:
        categories = ["low", "high"]

    class Landscape:
        fitness_layers = {"score": CategoryLayer()}

        def __init__(self):
            self.kwargs = None

        def to_sequence_tensors(self, **kwargs):
            self.kwargs = kwargs
            return [
                {
                    "sequence_tensor": torch.tensor([1, 2]),
                    "embedding": torch.tensor([0.5]),
                    "attention_mask": torch.tensor([1, 1]),
                    "fitness_tensors": {
                        "score": torch.tensor([1.0]),
                        "ignored": torch.tensor([2.0]),
                    },
                }
            ]

    landscape = Landscape()
    exported = export_landscape_records(
        landscape,
        fitness_layers=["score"],
        rename_fitness={"score": "target"},
        sequence_idx=[0],
        feature_view="embedding",
    )

    assert list(exported.records[0]["fitness_tensors"]) == ["target"]
    assert "embedding" in exported.records[0]
    assert "attention_mask" in exported.records[0]
    assert exported.fitness_mappings == {"target": ["low", "high"]}
    assert landscape.kwargs["sequence_idx"] == [0]
    assert landscape.kwargs["feature_view"] == "embedding"
    assert landscape.kwargs["as_batch"] is False


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"not": "a sequence"}, "must return a sequence"),
        (["not a mapping"], "must yield mapping records"),
        ([{"sequence_tensor": torch.tensor([1])}], "fitness_tensors mapping"),
    ],
)
def test_export_landscape_records_validates_return_contract(raw, message):
    landscape = SimpleNamespace(to_sequence_tensors=lambda **kwargs: raw)

    with pytest.raises(ValueError, match=message):
        export_landscape_records(landscape)


def test_export_landscape_records_validates_landscape_and_requested_layers():
    with pytest.raises(ValueError, match="must implement"):
        export_landscape_records(object())

    landscape = SimpleNamespace(
        to_sequence_tensors=lambda **kwargs: [
            {
                "sequence_tensor": torch.tensor([1]),
                "fitness_tensors": {"present": torch.tensor([1.0])},
            }
        ]
    )
    with pytest.raises(ValueError, match="missing from landscape export"):
        export_landscape_records(landscape, fitness_layers=["missing"])


def test_export_landscape_records_accepts_empty_export():
    landscape = SimpleNamespace(
        to_sequence_tensors=lambda **kwargs: [],
        fitness_layers={},
    )

    exported = export_landscape_records(landscape)

    assert exported.records == []
    assert exported.fitness_mappings == {}


def test_embedding_input_adapter_computes_missing_embeddings():
    class Landscape:
        active_embedding_domain = "plm"

        def __init__(self):
            self.embedding = None
            self.model_name = None

        def get_embedding(self):
            return self.embedding

        def get_embedding_metadata(self, domain):
            return {"model_name": "metadata-model", "embedding_mode": "hard"}

        def compute_plm_embeddings(self, *, model_name):
            self.model_name = model_name
            self.embedding = np.ones((2, 3), dtype=np.float32)

    landscape = Landscape()
    adapter = EmbeddingInputAdapter()

    batches = list(adapter.iter_batches(landscape, batch_size=8))

    assert landscape.model_name == "metadata-model"
    assert batches[0][0].shape == (2, 3)
    assert adapter.metadata(landscape)["embedding_mode"] == "hard"


def test_embedding_input_adapter_reports_unavailable_or_failed_computation():
    no_compute = SimpleNamespace(get_embedding=lambda: None)
    with pytest.raises(RuntimeError, match="cannot compute embeddings"):
        list(EmbeddingInputAdapter().iter_batches(no_compute, batch_size=1))

    failed = SimpleNamespace(
        get_embedding=lambda: None,
        compute_plm_embeddings=lambda **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="automatic computation failed"):
        list(EmbeddingInputAdapter().iter_batches(failed, batch_size=1))


def test_empty_embedding_and_node_index_inputs_yield_no_batches():
    empty_embedding = DummyLandscape(np.empty((0, 3), dtype=np.float32))
    assert (
        list(EmbeddingInputAdapter().iter_batches(empty_embedding, batch_size=2)) == []
    )
    assert (
        list(
            NodeIndexInputAdapter().iter_batches(
                SimpleNamespace(sequences=[]),
                batch_size=2,
            )
        )
        == []
    )


def test_prob_categorical_output_adapter_validates_outputs_and_categories():
    adapter = ProbCategoricalOutputAdapter()
    with pytest.raises(ValueError, match="requires 'mean'"):
        adapter.to_layer({}, None, {}, "prediction")
    with pytest.raises(ValueError, match="categories"):
        adapter.to_layer(
            {"mean": torch.ones(2, 3)},
            ["only", "two"],
            {},
            "prediction",
        )


def test_adapter_registries_reject_duplicates_and_invalid_output_adapters():
    class Model:
        pass

    class Adapter(EmbeddingInputAdapter):
        pass

    register_model_adapter(Model, lambda model: model, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_model_adapter(Model, lambda model: model)

    register_model_layer_mapping(Model, "numeric", overwrite=True)
    with pytest.raises(ValueError, match="already mapped"):
        register_model_layer_mapping(Model, "numeric")

    register_input_adapter("duplicate-test", Adapter, overwrite=True)
    with pytest.raises(ValueError, match="already registered"):
        register_input_adapter("duplicate-test", Adapter)

    from landscapyml.core.adaptor import register_output_adapter

    register_output_adapter("duplicate-output", NumericOutputAdapter, overwrite=True)
    with pytest.raises(ValueError, match="already exists"):
        register_output_adapter("duplicate-output", NumericOutputAdapter)
    with pytest.raises(TypeError, match="instance or subclass"):
        register_output_adapter("invalid-output", object(), overwrite=True)
    with pytest.raises(ValueError, match="No adapter registered"):
        resolve_output_adapter("missing-output")


def test_numeric_output_adapter_validates_and_detaches_gradients(monkeypatch):
    class StubFitness:
        @classmethod
        def from_tensor(cls, name, tensor, metadata):
            return SimpleNamespace(name=name, tensor=tensor, metadata=metadata)

    monkeypatch.setattr("landscapyml.core.adaptor.NumericFitness", StubFitness)
    adapter = NumericOutputAdapter()
    source = torch.tensor([1.0, 2.0], requires_grad=True)

    layer = adapter.to_layer({"prediction": source}, None, {}, "prediction")

    assert layer.tensor.device.type == "cpu"
    assert layer.tensor.requires_grad is False
    with pytest.raises(TypeError, match="non-tensor"):
        adapter.to_layer({"output": [1.0]}, None, {}, "prediction")
    with pytest.raises(ValueError, match="single tensor"):
        adapter.to_layer({}, None, {}, "prediction")
    with pytest.raises(ValueError, match="1-D or 2-D"):
        adapter.to_layer({"output": torch.zeros(1, 1, 1)}, None, {}, "prediction")
