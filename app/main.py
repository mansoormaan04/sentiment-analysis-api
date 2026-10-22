"""
Sentiment Analysis REST API
===========================

A small FastAPI service that classifies the sentiment of English text as
``positive``, ``negative`` or ``neutral`` (with a confidence score).

Two sentiment engines are supported, selected with the ``SENTIMENT_ENGINE``
environment variable:

* ``auto`` (default) - try the HuggingFace transformer model first and
  fall back to the lexicon engine if transformers/torch are unavailable
  or the model cannot be loaded.
* ``transformer`` - force the HuggingFace transformer model
  (``cardiffnlp/twitter-roberta-base-sentiment-latest``, a RoBERTa model
  fine-tuned for 3-way sentiment). Requires ``transformers`` and ``torch``
  (see the "Transformer engine" section of the README).
* ``vader`` - force the lexicon-based VADER engine (``vaderSentiment``).
  No model download, tiny footprint, works fully offline.

Endpoints
---------
GET  /health         -> service status + active engine
POST /predict        -> sentiment for a single text
POST /predict/batch  -> sentiment for a list of texts
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

ENGINE_SETTING = os.getenv("SENTIMENT_ENGINE", "auto").lower()
TRANSFORMER_MODEL = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# VADER compound-score thresholds (standard values from the VADER paper).
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

MAX_BATCH_SIZE = 100
MAX_TEXT_LENGTH = 5000


# ---------------------------------------------------------------------------
# Sentiment engines
# ---------------------------------------------------------------------------

class VaderEngine:
    """Lexicon-based sentiment using VADER (vaderSentiment)."""

    name = "vader"
    model = "vaderSentiment (lexicon + rules)"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> dict:
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= POS_THRESHOLD:
            label = "positive"
            confidence = compound
        elif compound <= NEG_THRESHOLD:
            label = "negative"
            confidence = abs(compound)
        else:
            label = "neutral"
            # The closer the compound score is to 0, the more confidently
            # the text is neutral.
            confidence = 1.0 - abs(compound)
        return {
            "sentiment": label,
            "confidence": round(float(confidence), 4),
            "scores": {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "compound": compound,
            },
        }


class TransformerEngine:
    """Transformer-based sentiment using a HuggingFace pipeline."""

    name = "transformer"

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self.model = model_name
        self._pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            top_k=None,  # return scores for every label
            truncation=True,
        )

    def predict(self, text: str) -> dict:
        raw = self._pipe(text)
        # With top_k=None the pipeline returns [[{label, score}, ...]] for
        # a single input.
        entries = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {}
        for entry in entries:
            label = str(entry["label"]).lower()
            # Normalise label names across models ("LABEL_0", "positive", ...)
            if label in ("label_0", "negative"):
                key = "negative"
            elif label in ("label_1", "neutral"):
                key = "neutral"
            elif label in ("label_2", "positive"):
                key = "positive"
            else:
                key = label
            scores[key] = round(float(entry["score"]), 4)
        best = max(scores, key=scores.get)
        return {
            "sentiment": best,
            "confidence": scores[best],
            "scores": scores,
        }


def load_engine() -> object:
    """Load the sentiment engine selected by SENTIMENT_ENGINE.

    In ``auto`` mode the transformer engine is attempted first and any
    failure (missing packages, no network, model error) falls back to
    VADER so the API always starts.
    """
    if ENGINE_SETTING == "vader":
        return VaderEngine()
    if ENGINE_SETTING == "transformer":
        return TransformerEngine(TRANSFORMER_MODEL)
    # auto
    try:
        return TransformerEngine(TRANSFORMER_MODEL)
    except Exception as exc:  # noqa: BLE001 - fallback is the whole point
        print(f"[startup] Transformer engine unavailable ({exc!r}); "
              "falling back to VADER lexicon engine.")
        return VaderEngine()


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH,
                      examples=["I absolutely love this product!"])


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE,
                             examples=[["I love this!", "This is terrible.",
                                        "The package arrived on Tuesday."]])


class Prediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict


class BatchResponse(BaseModel):
    results: List[Prediction]
    count: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = load_engine()
    print(f"[startup] Active sentiment engine: {state['engine'].name} "
          f"({state['engine'].model})")
    yield
    state.clear()


app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text sentiment (positive / negative / neutral) "
                "with a transformer model or a VADER lexicon fallback.",
    version="1.0.0",
    lifespan=lifespan,
)


def _predict_one(text: str) -> Prediction:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422,
                            detail="Text must not be empty or whitespace.")
    result = state["engine"].predict(text)
    return Prediction(text=text, **result)


@app.get("/health")
def health() -> dict:
    engine = state.get("engine")
    return {
        "status": "ok",
        "engine": engine.name if engine else "loading",
        "model": engine.model if engine else None,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Sentiment Analysis API",
        "docs": "/docs",
        "endpoints": {
            "GET /health": "service status and active engine",
            "POST /predict": "sentiment for one text: {\"text\": \"...\"}",
            "POST /predict/batch": "sentiment for many texts: "
                                   "{\"texts\": [\"...\", \"...\"]}",
        },
    }


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest) -> Prediction:
    return _predict_one(request.text)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    results = [_predict_one(t) for t in request.texts]
    return BatchResponse(results=results, count=len(results))
