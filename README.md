# Sentiment Analysis REST API

A lightweight REST API that classifies the sentiment of English text as
**positive**, **negative**, or **neutral**, with a confidence score.

Built with **FastAPI**, with two interchangeable sentiment engines selected
by the `SENTIMENT_ENGINE` environment variable (`auto` / `transformer` /
`vader`):

| Engine | How it works | Notes |
|---|---|---|
| `transformer` | HuggingFace pipeline with `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa fine-tuned for 3-way sentiment) | Best accuracy; needs `transformers` + `torch` (~2 GB) and a one-time ~500 MB model download |
| `vader` | VADER lexicon + rule-based valence (`vaderSentiment`) | Tiny, fast, fully offline; used automatically when the transformer stack is not installed |

In `auto` mode (the default) the API tries the transformer first and falls
back to VADER if it cannot be loaded, so the service always starts.

> **Verified end-to-end in both modes** — every example response below is a
> real output captured from the running service (see *Verified test
> results*).

---

## Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```

Transformer engine active:

```json
{"status": "ok", "engine": "transformer", "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"}
```

VADER fallback active (e.g. minimal/Docker install):

```json
{"status": "ok", "engine": "vader", "model": "vaderSentiment (lexicon + rules)"}
```

### `POST /predict` — single text

Request body: `{"text": "..."}` → response with `sentiment`, `confidence`
and a per-class `scores` breakdown. Examples in *Verified test results*.

### `POST /predict/batch` — multiple texts

Request body: `{"texts": ["...", "..."]}` (max 100 texts) → `{"results": [...], "count": n}`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Verified test results

Captured with `uvicorn app.main:app` + `curl` against the live service.

### Transformer engine (`SENTIMENT_ENGINE=auto`, full install)

**Positive**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product! The quality is amazing and it made me really happy."}'
```

```json
{"text": "I absolutely love this product! The quality is amazing and it made me really happy.", "sentiment": "positive", "confidence": 0.9896, "scores": {"positive": 0.9896, "neutral": 0.006, "negative": 0.0044}}
```

**Negative**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "This was a terrible experience. The service was awful and I am very disappointed."}'
```

```json
{"text": "This was a terrible experience. The service was awful and I am very disappointed.", "sentiment": "negative", "confidence": 0.9514, "scores": {"negative": 0.9514, "neutral": 0.0421, "positive": 0.0065}}
```

**Neutral**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "The package arrived on Tuesday and contains three items."}'
```

```json
{"text": "The package arrived on Tuesday and contains three items.", "sentiment": "neutral", "confidence": 0.8461, "scores": {"neutral": 0.8461, "positive": 0.1496, "negative": 0.0043}}
```

**Batch**

```bash
curl -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" \
  -d '{"texts": ["What a fantastic day, everything went perfectly!", "I hate this, it is the worst purchase I have ever made.", "The meeting is scheduled for 10 AM tomorrow."]}'
```

```json
{"results": [{"text": "What a fantastic day, everything went perfectly!", "sentiment": "positive", "confidence": 0.989, "scores": {"positive": 0.989, "neutral": 0.0065, "negative": 0.0045}}, {"text": "I hate this, it is the worst purchase I have ever made.", "sentiment": "negative", "confidence": 0.9548, "scores": {"negative": 0.9548, "neutral": 0.0387, "positive": 0.0065}}, {"text": "The meeting is scheduled for 10 AM tomorrow.", "sentiment": "neutral", "confidence": 0.9481, "scores": {"neutral": 0.9481, "positive": 0.0417, "negative": 0.0102}}], "count": 3}
```

### VADER engine (`SENTIMENT_ENGINE=vader`, minimal install)

Same requests against the lexicon engine:

| Text | Result |
|---|---|
| "I absolutely love this product! The quality is amazing and it made me really happy." | `positive`, confidence **0.9271** (compound 0.9271) |
| "This was a terrible experience. The service was awful and I am very disappointed." | `negative`, confidence **0.8588** (compound -0.8588) |
| "The package arrived on Tuesday and contains three items." | `neutral`, confidence **1.0** (compound 0.0) |
| Batch: fantastic day / worst purchase / meeting at 10 AM | `positive` 0.8439 · `negative` 0.8316 · `neutral` 1.0 |

### How labels are decided (VADER engine)

VADER produces a `compound` valence score in [-1, 1]. Standard thresholds
are applied: `compound >= 0.05` → positive, `compound <= -0.05` →
negative, otherwise neutral. Confidence is `|compound|` for
positive/negative and `1 - |compound|` for neutral. The transformer engine
returns the model's softmax score for the predicted class.

---

## Run locally

Requires Python 3.10+.

**Minimal (VADER engine):**

```bash
git clone <this-repo>
cd sentiment-analysis-api

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Full (transformer engine):** additionally install the transformer stack,
then start the same way — `auto` mode picks it up automatically:

```bash
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --host 0.0.0.0 --port 8000   # model downloads on first start
```

Force a specific engine or model:

```bash
SENTIMENT_ENGINE=vader uvicorn app.main:app --port 8000
SENTIMENT_ENGINE=transformer SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english \
  uvicorn app.main:app --port 8000
```

## Run with Docker

```bash
docker build -t sentiment-analysis-api .
docker run -p 8000:8000 sentiment-analysis-api
```

The Docker image installs only the core `requirements.txt`, so it stays
small and serves the VADER engine (`SENTIMENT_ENGINE=auto` → VADER
fallback). To serve the transformer engine from Docker, add the optional
install line above to the Dockerfile and set `-e SENTIMENT_ENGINE=transformer`.

## Tech stack

- **FastAPI** + **Uvicorn** — API framework and ASGI server
- **Pydantic v2** — request/response validation
- **Transformers + PyTorch** — RoBERTa sentiment model
  (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **VADER** (`vaderSentiment`) — lexicon-based fallback engine
- **Docker** — `python:3.12-slim` image

## Project structure

```
sentiment-analysis-api/
├── app/
│   ├── __init__.py
│   └── main.py          # API endpoints + both sentiment engines
├── requirements.txt     # core deps (transformer deps documented inside)
├── Dockerfile
├── .gitignore
└── README.md
```
