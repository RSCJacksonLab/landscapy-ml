# CLI

The CLI is exposed via `python -m landscapyml`.

## Commands
- `list`: print registered model keys, data builders, and landscape runners.
- `train-landscape`: train a landscape-regression model from CSV split files.

## Examples
```bash
python -m landscapyml list

python -m landscapyml train-landscape \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz \
  --model-key graph_attention_regressor \
  --data-name landscape_graph_regression \
  --max-epochs 1 \
  --data-kwargs '{"normalize_features": true}' \
  --trainer-kwargs '{"use_wandb": false, "checkpoint_monitor": null}'

python -m landscapyml train-landscape \
  --csv-path demo/rhomax/by_wild_type/by_wild_type.csv.gz \
  --model-key diffusion_prior_gp \
  --fit-kwargs '{"training_iters": 2, "learning_rate": 0.05}'
```

When an input has a split column, every non-validation row must match the
configured train or test label after trimming whitespace and normalizing case.
Blank or unknown labels are rejected with their CSV row numbers; validation
markers take precedence. Result JSON records train, validation, test, and total
assigned row counts.
