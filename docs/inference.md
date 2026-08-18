# Inference and landscape integration

Inference helpers live in `landscapyml.core.inference` and operate on trained models plus either raw sequences, FitnessLandscape records, or existing embeddings. Adapter ABCs and registries live in `landscapyml.core.adaptor`.

## Direct sequence predictions
- `predict_sequences(model, sequences, *, embedding_mode="hard", model_name="facebook/esm2_t6_8M_UR50D", device=None, embedding_batch_size=32) -> (mean_probs, variance)`
  - Embeds raw sequences via `embed_sequences` (no tokens returned) and runs `model.predict_with_uncertainty`.
  - The model must implement `predict_with_uncertainty`.

## FitnessLandscape record predictions
- `predict_landscape_records(model, records) -> (mean_probs, variance)`
  - Accepts iterable of record dicts (from `FitnessLandscape.to_sequence_tensors(as_batch=False)` or produced by `embed_sequences_to_records`).
  - Uses `embedding` if present, else `sequence_tensor`, stacks tensors, and forwards to `predict_with_uncertainty`.

## FitnessLandscape layer attachment
- `infer_fitness_layer_from_landscape(landscape, model, *, batch_size=256, num_workers=0, device=None, attach=True, inplace=True, layer_name="predicted_fitness", categories=None, input_adapter=None, input_adapter_kwargs=None)`
  - Resolves a model adapter for `model` (registered or provided) to standardize inference.
  - Resolves a landscape input adapter (defaults to embedding-based extraction).
  - Ensures embedding domain/model compatibility between the adapter and landscape when available.
  - Batches inputs through the adapter and builds a `ProbabilisticCategoricalFitness` layer via the registered output adapter. When `attach` is true, attaches to the landscape (copying if `inplace` is false).
  - The `fitness_landscape` dependency is optional; when absent, imports fall back to stub types and landscape-related helpers are unavailable.

## Extensibility
- `LandscapeInputAdapter`: ABC that yields batches from a landscape and converts them into model inputs.
- `GraphTensorInputAdapter`: generic core adapter for models that consume `landscape.to_graph_tensor(...)`; it falls back to graph edges plus embeddings or length/composition node features when one-hot graph tensors cannot represent variable-length sequences.
- `NodeIndexInputAdapter`: generic core adapter for models that consume landscape node indices as inputs.
- `LandscapeOutputAdapter`: ABC that converts model outputs into landscape fitness layers.
- `register_input_adapter(name, factory, overwrite=False)`: Register a landscape input adapter (e.g., graph/structure extractors).
- `register_model_adapter(model_cls, adapter_factory, overwrite=False)`: Register a model adapter for a model class. Adapters expose a `layer_kind` and a `predict(inputs)` method.
- `register_model_layer_mapping(model_cls, layer_kind, overwrite=False)`: Map a model class and its subclasses to a logical layer kind string (e.g., `prob_categorical`) when the default adapter is sufficient.
- `register_output_adapter(kind, adapter, overwrite=False)`: Register a landscape output adapter class/instance for a layer kind.
- `register_layer_adapter(kind, adapter, overwrite=False)`: Convenience wrapper for function-style output adapters.

Model-specific bridges should live at the package edge rather than in the
shared core.

Both model adapter factories and model-to-layer mappings are resolved using
Python's method resolution order (MRO). An exact-class registration therefore
wins over a base-class registration; with multiple inheritance, the first
registered class in the model's MRO wins. A registered adapter factory takes
priority over a model-to-layer mapping because it defines the complete
inference interface.

For graph-native models, `landscapyml.examples.gat_fitness` reuses the core
`GraphTensorInputAdapter` and also exposes a backwards-compatible
`landscape_graph` alias:
- `attach_graph_attention_predictions(...)` as a convenience wrapper around
  `infer_fitness_layer_from_landscape(...)`
- a `GraphAttentionFitnessRegressor` example whose outputs attach as a numeric
  fitness layer

For diffusion-prior GP workflows, `landscapyml.examples.gp_fitness` reuses the
core `NodeIndexInputAdapter` and also exposes a backwards-compatible
`landscape_node_index` alias:
- `attach_diffusion_gp_predictions(...)` as a convenience wrapper around
  `infer_fitness_layer_from_landscape(...)`
- `DiffusionPriorExactGP`, which predicts numeric fitness values for the nodes
  already present in a fixed landscape graph

Example: registering a new model and adapter for a density-style output
```python
from landscapyml.core.adaptor import register_model_layer_mapping, register_layer_adapter
from fitness_landscape.core.fitness import ContinuousFitness

class MyModel:
    def predict(self, x): ...

def my_layer_adapter(outputs, categories, metadata, layer_name):
    return ContinuousFitness(
        name=layer_name,
        values=outputs["mean"].numpy(),
        metadata=metadata,
    )

register_model_layer_mapping(MyModel, "continuous_density")
register_layer_adapter("continuous_density", my_layer_adapter)
```

Returned tensors are moved to CPU for downstream use; callers can re-map to other devices as needed.

Example: custom model adapter for non-standard predict signatures
```python
from landscapyml.core.adaptor import register_model_adapter

class MyAdapter:
    layer_kind = "prob_categorical"

    def __init__(self, model):
        self.model = model

    def predict(self, inputs):
        mean, var = self.model.infer(inputs)
        return {"mean": mean, "var": var}

register_model_adapter(MyExternalModel, lambda model: MyAdapter(model))
```
