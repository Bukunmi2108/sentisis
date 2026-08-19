# 05 — Making inference fast

**Status:** Accepted, pending measurement. The latency and F1 comparison is filled in after export.

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
