# 02 — Two models, and how I pick between them

**Status:** Accepted. DistilBERT won on macro-F1, 0.6737 against the baseline's 0.5977.

## Context

I want two models rather than one: a cheap classical baseline that sets the bar, and a
transformer that has to clear it to justify its cost. Only the winner gets served.

"Best" needs defining up front, because on an imbalanced problem different metrics pick
different winners.

## Decision

**Baseline:** TF-IDF (unigrams and bigrams) into multinomial logistic regression.

**Transformer:** `distilbert-base-uncased`, fully fine-tuned. Max length 96, 3 epochs,
learning rate 2e-5, batch size 32.

Max length is 96 because measured wordpiece lengths are p50 30, p95 41, p99 47. It truncates
0.02% of test; 128 would only pad further.

**The winner is whichever has the higher macro-F1 on the test split.** I fixed this rule
before training either model.