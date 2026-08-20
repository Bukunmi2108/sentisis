# 07 — Shipping the model file

**Status:** Accepted

## Context

A fresh clone has to be able to make predictions immediately, with no setup step beyond
`make up`. But the model files are large, and GitHub rejects any single file over 100 MB.

## Decision

Commit to git:

- `model/artifacts/distilbert/model_int8.onnx` — the quantised model I serve, 65 MB
- `model/artifacts/baseline.joblib` — the TF-IDF and logistic regression pipeline, 5.7 MB
- `model/artifacts/distilbert/tokenizer/` — small

The full-precision checkpoint (265 MB) is **not** committed and is not published. Nothing here
needs it — the fp32 scores it is compared against are in `training_summary.json`, and the Kaggle
notebook reproduces it from a fixed seed.

## Why

This is the only option where `git clone` followed by `make up` works with no network access, no
extra step, and no extra tooling installed on the machine doing the cloning.

Quantisation ([note 05](05-making-inference-fast.md)) is what makes it fit. 65 MB sits well under
the limit, so the size problem is solved by a decision I'd already made for speed reasons.
