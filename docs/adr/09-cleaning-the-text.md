# 09 — Cleaning the text

**Status:** Accepted. Whether TweetEval pre-normalises mentions and links is verified during data
preparation, not assumed here.

## Context

The cleaning rules are: turn emoji into text tags, replace mentions with `[user]` and links with
`[url]`, and keep only letters and spaces for TF-IDF.

Applied to everything, that last rule would also strip capitals and punctuation from the
DistilBERT input. But `WHAT` and `!!!` carry sentiment, and a transformer's tokenizer is built to
use them. Stripping them would hold back the stronger model to suit the weaker one.

## Decision

Two stages, both pure functions in `model/preprocess.py`, used by training **and** serving.

**Stage A — `normalize(text)`**, used by both models and by the API:

1. Unicode NFKC normalisation
2. Links become `[url]`
3. Mentions become `[user]`
4. Emoji become text tags, written as `[fire]`
5. Collapse repeated whitespace and strip

**Stage B — `to_bow(text)`**, TF-IDF only, applied after stage A:

6. Lowercase
7. Keep letters and spaces only
8. Collapse whitespace and strip

DistilBERT gets stage A. TF-IDF gets stage B.