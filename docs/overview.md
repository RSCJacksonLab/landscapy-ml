# landscapy-ml Overview

landscapy-ml provides a generic interface layer between `landscapy`
`FitnessLandscape` objects and ML pipelines. The package centers around
small, composable helpers for exporting landscapes into records, wiring those
records into datasets and Lightning datamodules, and adapting model outputs
back into landscape fitness layers.

Model-specific integrations are examples built on top of that core rather than
the primary abstraction.

## Package layout
- `landscapyml.core.data`: Base landscape record datasets/data modules plus the graph-regression data path used by the demos.
- `landscapyml.core.data_utils`: Record normalization, target/input getters, sequence embedding helpers, split helpers, and variable-length-safe sequence features.
- `landscapyml.core.adaptor`: The landscapy <-> PyTorch format boundary: landscape record export, input adapters, model adapters, and output-layer adapters.
- `landscapyml.core.model_registry`: Functional model/data registration. This is what the CLI and `TrainingJob` use to resolve model and data names.
- `landscapyml.core.trainer`: Trainer construction and `TrainingJob`. It consumes the registry but does not own it.
- `landscapyml.core.inference`: Trained model -> `FitnessLandscape` mapping, including attaching predicted fitness layers.
- `landscapyml.examples`: Optional model-specific examples for the maintained GAT and diffusion-prior GP demo paths.
- `landscapyml.logging_utils`: Lightweight logging configuration aligned with landscapy.
- `landscapyml.__main__`: CLI entry point (`python -m landscapyml`).

Use the CLI for demo training, or import the Python APIs for programmatic training/inference. Other docs in this folder dive into specific topics.
