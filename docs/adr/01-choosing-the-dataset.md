# 01 — Choosing the dataset

**Status:** Accepted

## Context

The API has to return three labels: positive, negative, and neutral. Most well-known sentiment
datasets (IMDB, Yelp polarity) only have two. Using one of them would mean inventing a neutral
class that the data doesn't actually contain.

## Decision

Use **`tweet_eval/sentiment`** with its published train/validation/test splits, unchanged.
Labels are `0 = negative`, `1 = neutral`, `2 = positive`.

## Why

It has three real classes, labelled by human annotators. Nothing is invented.

The text is also the right kind of hard. Tweets are short and messy, full of emoji, mentions,
and links, so the cleaning work in [note 09](09-cleaning-the-text.md) matters instead of being
decorative. That messiness is also where a transformer should beat a bag-of-words model, which
makes the comparison in [note 02](02-two-models.md) worth running.