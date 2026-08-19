# 04 — Training on a rented GPU

**Status:** Accepted

## Context

The development machine has no GPU: 4 CPU cores and 15 GB of RAM. Fine-tuning DistilBERT on
roughly 45,000 tweets here was estimated at 4 to 6 hours. The whole project has a 3-day budget.

## Decision

The training code lives in `model/distilbert.py` as a normal importable module.
`model/notebooks/` holds a thin notebook that installs dependencies, pulls the repo, and calls
into that module. It contains no training logic of its own.

The notebook runs on a free Colab or Kaggle T4, which takes about 10 minutes. The trained model
comes back into the repo as described in [note 07](07-shipping-the-model-file.md).