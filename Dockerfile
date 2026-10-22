FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Default engine is "auto": without transformers installed this resolves
# to the VADER lexicon engine, so the image stays small and starts fast.
# To use the transformer engine instead, install the optional deps (see
# README) and set:  docker run -e SENTIMENT_ENGINE=transformer ...
ENV SENTIMENT_ENGINE=auto

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
