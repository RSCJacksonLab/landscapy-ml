# API Reference

## Core Data
- `LandscapeDataset`, `LandscapeDataModule`
- `LandscapeGraphDataset`, `LandscapeGraphRegressionDataModule`
- `build_regression_graph_from_landscape`
- `make_preferred_input_getter`, `make_fitness_target_getter`

## Adapters
- `LandscapeExport`, `export_landscape_records`
- `LandscapeInputAdapter`, `GraphTensorInputAdapter`, `NodeIndexInputAdapter`
- `ModelAdapter`
- `LandscapeOutputAdapter`
- `register_input_adapter`, `register_model_adapter`, `register_model_layer_mapping`
- `register_output_adapter`, `register_layer_adapter`

## Registry And Training
- `register_model`, `register_data`
- `available_models`, `available_data_builders`
- `TrainingJob`, `create_trainer`

## Inference
- `predict_sequences`
- `predict_landscape_records`
- `infer_fitness_layer_from_landscape`

## Demos
- `landscapyml.examples.gat_fitness`
- `landscapyml.examples.gp_fitness`
- `python -m landscapyml list`
- `python -m landscapyml train-landscape`
