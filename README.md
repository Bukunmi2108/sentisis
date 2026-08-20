# sentisis

A sentiment analysis API. You send it text, it tells you whether the text is negative, neutral or
positive, and how confident it is.

**Live: https://sentisis.duckdns.org**

```bash
curl -X POST https://sentisis.duckdns.org/predict \
  -H 'content-type: application/json' \
  -d '{"text": "i love this 🔥"}'
```

```json
{
  "label": "positive",
  "confidence": 0.9809,
  "probabilities": {"negative": 0.0062, "neutral": 0.0129, "positive": 0.9809},
  "model_version": "distilbert-int8-v1",
  "latency_ms": 14.8,
  "cached": false
}
```

## The models, and which one I serve

I trained two models on the TweetEval sentiment dataset and scored both on the same held-out test
split of 12,284 tweets.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + logistic regression | 0.6018 | 0.5977 |
| **DistilBERT** | **0.6730** | **0.6737** |

I decided before training that the winner would be whichever scored higher **macro-F1**, because
the classes are imbalanced and plain accuracy would reward a model that simply guesses the biggest
class. DistilBERT won by 7.6 points, so that is what the API serves.

Both models are trained with balanced class weights. This helps the model notice the smaller
classes: negative recall went from 0.34 to 0.82 on the baseline. The cost is that neutral text
often gets pushed into one of the two poles, which is why neutral is the weakest class.

Before serving, I converted DistilBERT to ONNX and quantised it to 8-bit integers. That shrank the
file from 265 MB to 65 MB and made it small enough to commit, so a fresh clone can predict
immediately with no download. Accuracy barely moved: macro-F1 went from 0.6755 to 0.6737, a loss
of 0.0018.

The API container therefore needs neither PyTorch nor transformers, only ONNX Runtime and a
tokeniser. The image is 422 MB instead of several gigabytes.

Longer reasoning for each of these choices is in [`docs/adr/`](docs/adr/), and what the data
actually looks like is in [`docs/findings/01-data.md`](docs/findings/01-data.md).

## Setup

You need Python 3.11 or newer. The trained model is in the repository, so nothing is downloaded.

To run the API:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To also train models or run the tests, install the extras instead:

```bash
pip install -e ".[dev,train]"
```

## Running the API

Locally:

```bash
make serve
```

With Docker, which also starts Redis:

```bash
make up
```

Either way it is on http://localhost:8000. The root page is a small demo you can type into, and
http://localhost:8000/docs is the interactive API documentation.

### Endpoints

| Endpoint | What it does |
|---|---|
| `POST /predict` | classify one text |
| `POST /predict/batch` | classify up to 128 texts, results come back in the order you sent them |
| `GET /health` | is the service running |
| `GET /ready` | is the model loaded and working |

`/health` and `/ready` answer different questions. `/health` says the process is alive and touches
nothing. `/ready` actually runs a small prediction, so it proves the model works.

### When something is wrong

Every error looks the same, with a code you can check in your own code and an id you can quote to
me:

```json
{
  "error": {
    "code": "text_too_long",
    "message": "text exceeds the 5000 character limit",
    "retryable": false,
    "request_id": "023e5b8026594f68945ec3232850ea18"
  }
}
```

Empty text, text over 5,000 characters, and a batch that is empty or over 128 items all return
422. If one text in a batch is invalid, the whole batch is rejected rather than half answered. If
the model is not loaded you get 503, and `retryable` tells you whether trying again might help.

### Caching

Repeated text is served from Redis instead of being run through the model again, which takes a
request from about 90 ms to about 4 ms. The response says `"cached": true` when this happens.

If Redis is down, the API logs it and runs the model anyway. A cache failure never fails a
request, and there is a test that stops Redis and proves it.

## Training

```bash
make data     # download the dataset
make train    # train the baseline and write its scores
make eval     # print both models side by side
```

`make train` runs `model/notebooks/train_baseline.ipynb` from top to bottom and saves the outputs
back into the notebook, so what you see in the repository is what actually ran.

DistilBERT is different because I have no GPU on my machine. Its notebook,
`model/notebooks/train_distilbert.ipynb`, is written to run standalone on Kaggle and carries its
own copy of the cleaning code. I run it there, download the result into
`model/artifacts/distilbert/`, and score it locally with:

```bash
python -m model.scripts.score_distilbert
```

## Tests

```bash
make test               # everything
make test-unit          # no Redis needed
make test-integration   # needs Redis: make redis
make lint               # formatting, linting and strict type checking
```

There are 114 tests. The integration tests run against a real Redis rather than a fake, because a
fake that slowly drifts from how Redis really behaves is exactly how this kind of thing breaks
later.

## How the code is arranged

```
api/        the web service: routes, validation, errors, model loading, caching
model/      the data, the two models, the metrics, and the text cleaning
web/        the demo page
tests/      unit tests and integration tests
docs/adr/   why each decision was made
```

The one piece worth pointing out is `model/preprocess.py`. Both the training code and the API
import it, so text is cleaned at prediction time exactly as it was during training.

## Deployment

Pushing to `main` runs the tests, builds the image, and if everything passes, deploys that exact
commit to the server behind https://sentisis.duckdns.org. The workflow is in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).
