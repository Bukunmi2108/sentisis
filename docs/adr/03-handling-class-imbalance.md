# 03 — Handling class imbalance

**Status:** Accepted. Actual class counts get printed during data preparation.

## Context

The three sentiment classes are not evenly represented. Trained naively, a model gets a good
accuracy score by over-predicting the common class and largely ignoring the rare one. Since I
select on macro-F1, this directly decides which model wins.

## Decision

Weight the loss by inverse class frequency, in both models:

- Baseline: `class_weight='balanced'` on `LogisticRegression`.
- DistilBERT: a `Trainer` subclass that passes class weights to `CrossEntropyLoss`.

Weights are computed from the training split only.