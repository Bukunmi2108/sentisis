# 09 — Cleaning the text

**Status:** Accepted. Step 1 was added after measuring the data — see
[finding 01](../findings/01-data.md).

## Context

The cleaning rules are: turn emoji into text tags, replace mentions with `[user]` and links with
`[url]`, and keep only letters and spaces for TF-IDF.

Applied to everything, that last rule would also strip capitals and punctuation from the
DistilBERT input. But `WHAT` and `!!!` carry sentiment, and a transformer's tokenizer is built to
use them. Stripping them would hold back the stronger model to suit the weaker one.

## Decision

Two stages, both pure functions in `model/preprocess.py`, used by training **and** serving.

**Stage A — `normalize(text)`**, used by both models and by the API:

1. Expand HTML entities and escaped unicode, repeatedly until the text stops changing
2. Unicode NFKC normalisation
3. Links become `[url]`
4. Mentions become `[user]`
5. Emoji become text tags, written as `[fire]`
6. Collapse repeated whitespace and strip

**Stage B — `to_bow(text)`**, TF-IDF only, applied after stage A:

7. Lowercase
8. Keep letters and spaces only
9. Collapse whitespace and strip

DistilBERT gets stage A. TF-IDF gets stage B.
