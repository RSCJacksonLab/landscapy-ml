# Inference and landscape integration

Inference helpers live in `landscapyml.inference` and operate on trained models plus either raw sequences, FitnessLandscape records, or existing embeddings.

## Direct sequence predictions
- `predict_sequences(model, sequences, *, embedding_mode="hard", model_name="facebook/esm2_t6_8M_UR50D", device=None, embedding_batch_size=32) -> (mean_probs, variance)`
  - Embeds raw sequences via `embed_sequences` (no tokens returned) and runs `model.predict_with_uncertainty`.
  - The model must implement `predict_with_uncertainty` (available on `SequenceGPClassifier` and `SequenceMLPEnsembleClassifier`).

## FitnessLandscape record predictions
- `predict_landscape_records(model, records) -> (mean_probs, variance)`
  - Accepts iterable of record dicts (from `FitnessLandscape.to_sequence_tensors(as_batch=False)` or produced by `embed_sequences_to_records`).
  - Uses `embedding` if present, else `sequence_tensor`, stacks tensors, and forwards to `predict_with_uncertainty`.

## FitnessLandscape layer attachment
- `infer_fitness_layer_from_landscape(landscape, model, *, batch_size=256, num_workers=0, device=None, attach=True, inplace=True, layer_name="predicted_fitness", categories=None)`
  - Determines the layer kind for the model via `_MODEL_TO_LAYER` mapping (defaults cover GP and MLP ensemble).
  - Ensures embedding domain/model compatibility between the model and landscape when available.
  - Computes embeddings on the landscape if missing (using `landscape.compute_plm_embeddings`).
  - Batches embeddings through the model and builds a `ProbabilisticCategoricalFitness` layer via the registered adapter. When `attach` is true, attaches to the landscape (copying if `inplace` is false).
  - The `fitness_landscape` dependency is optional; when absent, imports fall back to stub types and landscape-related helpers are unavailable.

## Extensibility
- `register_model_layer_mapping(model_cls, layer_kind, overwrite=False)`: Map additional model types to a logical layer kind string (e.g., `prob_categorical`). This controls how models are interpreted when attaching to a `FitnessLandscape`.
- `register_layer_adapter(kind, adapter, overwrite=False)`: Provide custom adapter functions that convert model outputs into landscape fitness layers. An adapter receives `(outputs_dict, categories, metadata_dict, layer_name)` and returns a `Fitness` layer.

Example: registering a new model and adapter for a density-style output
```python
from landscapyml.inference import register_model_layer_mapping, register_layer_adapter
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
