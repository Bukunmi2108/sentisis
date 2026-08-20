# 05 — Making inference fast

**Status:** Accepted and measured. Int8 ships.

## Context

The model is served on CPU, because the target host has no GPU. Running PyTorch and
`transformers` in the API container would pull in around 2 GB of dependencies just to run a
forward pass, using a framework built mainly for training.

## Decision

Export the winning model to **ONNX**, apply **dynamic int8 quantisation**, and serve it with
**ONNX Runtime**. Benchmark latency and macro-F1 for both the fp32 and int8 versions on the same
inputs.

**Acceptance rule, set before measuring:** ship int8 if its macro-F1 is within **1.0 point** of
fp32. If the drop is larger, ship fp32 ONNX instead and record what it cost.

## Why

The API container then only needs ONNX Runtime and a tokenizer instead of the full training
stack. That means a smaller image, a faster cold start, and far fewer dependencies in the thing
I actually deploy.

Int8 usually gives a solid CPU speedup and a roughly 4x smaller weight file. That smaller file
is also what makes [note 07](07-shipping-the-model-file.md) possible at all.

## Measured

Int8 test macro-F1 0.6737 against fp32's 0.6755 — a drop of **0.0018**, inside the 1.0-point
budget, so int8 ships. The file went from 265 MB to 65 MB.

Only 89.8% of individual labels agree, because the disagreements sit on low-confidence rows whose
logit margins are smaller than quantisation noise. They were as likely to be wrong either way, so
macro-F1 barely moves.

Latency on a 2-thread container over 100 requests: p50 52.5 ms and p95 131.7 ms on a cache miss,
p50 1.1 ms on a hit, 6.3 ms per text batched at 64.
