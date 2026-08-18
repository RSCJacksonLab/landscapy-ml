# Command-line interface

The CLI lists registered components and runs the maintained CSV training
workflows through `python -m landscapyml` or the `landscapy-ml` entry point.

## Commands

- `list` prints registered model keys, data builders, and landscape runners.
- `train-landscape` trains a registered landscape-regression workflow from CSV
  split files.

## Examples

List the registered components:

```bash
python -m landscapyml list
```

The repository includes a four-sequence portable CSV fixture. It can be used
to exercise the diffusion-prior GP workflow without downloading data:

```bash
python -m landscapyml train-landscape \
  --csv-path src/landscapyml/data/minimal_landscape.csv \
  --model-key diffusion_prior_gp \
  --fit-kwargs '{"training_iters": 2, "learning_rate": 0.05}'
```

When an input has a split column, every non-validation row must match the
configured train or test label after trimming whitespace and normalizing case.
Blank or unknown labels are rejected with their CSV row numbers. Validation
markers take precedence. Result JSON records train, validation, test, and total
assigned row counts.

The `train-landscape` workflow delegates CSV construction to Landscapy with a
Hamming graph. It preserves disconnected components and does not replace the
requested topology with a k-nearest-neighbor graph. Inputs that cannot form a
Hamming graph, including unaligned variable-length sequences, fail with the
Landscapy construction error.
