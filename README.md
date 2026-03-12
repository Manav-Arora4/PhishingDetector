# Real-Time AI/ML-Based Phishing Detection and Prevention System

Production-style Python project for phishing detection across message text, URLs, and webpage structure. The codebase is organized for local dataset training, synthetic phishing generation, incremental retraining, explainable inference, and async API serving.

## Highlights

- Local-only dataset ingestion from [`dataset/`](./dataset)
- Offline-safe training backend with optional HuggingFace/PyTorch transformer path
- Synthetic phishing text and URL generation
- Incremental retraining pipeline using synthetic samples and analyst feedback
- Async real-time API with `/analyze/text`, `/analyze/url`, `/analyze/full`, and `/feedback`
- SQLite-backed prediction logging and continuous learning simulation
- URL and HTML structural risk analysis
- Pytest suite covering dataset loading, training, retraining, inference, API behavior, and full-pipeline blocking

## Project Structure

```text
app/
  api/
  ml/
    training/
    synthetic/
    retraining/
  models/
  security/
  services/
  utils/
dataset/
tests/
train_base_model.py
retrain_incremental.py
README.md
pyproject.toml
```

## Architecture

### 1. Dataset Pipeline

The unified loader lives in `app/ml/training/load_local_datasets.py`.

It:

- scans every CSV in `dataset/`
- normalizes column names
- auto-detects text columns from `text`, `body`, `email`, `content`, `message`, and `text_combined`
- normalizes labels to `phishing=1` and `benign=0`
- combines subject and body content when useful
- removes duplicate messages

### 2. Training Backends

The training module supports two execution modes:

- `transformer`: uses PyTorch + HuggingFace Transformers when those dependencies and model assets are available
- `token_naive_bayes`: offline-safe fallback used in restricted environments and tests

Both backends expose the same interface for:

- `fit`
- `partial_fit`
- `predict_proba`
- `explain`
- `save/load`

Base model artifacts are written to:

- `app/models/phishing_detector.pt`
- `app/models/base_training_metrics.json`

### 3. Synthetic Data Generation

`app/ml/synthetic/generator.py` generates:

- urgency-heavy phishing messages
- AI-themed trust language
- enterprise credential prompts
- homoglyph domains
- deep subdomains
- base64-like query parameters

Generated files are saved under:

- `app/ml/synthetic/generated_data/synthetic_messages.jsonl`
- `app/ml/synthetic/generated_data/synthetic_urls.jsonl`
- `app/ml/synthetic/generated_data/synthetic_training_samples.jsonl`

### 4. Incremental Retraining

`app/ml/retraining/retrain_incremental.py`:

- loads the persisted model
- loads synthetic phishing samples
- reads analyst feedback from SQLite
- applies incremental updates
- compares metrics before and after retraining

Artifacts:

- `app/models/retraining_metrics.json`

### 5. Real-Time Security Analysis

The scoring pipeline combines:

- NLP phishing probability
- URL risk analysis
- structural webpage analysis

Outputs:

```json
{
  "final_risk_score": 0.91,
  "decision": "BLOCK",
  "explanation": {
    "nlp_score": 0.88,
    "url_score": 0.95,
    "structural_score": 0.62
  }
}
```

### 6. Explainability

The NLP service returns:

- `phishing_probability`
- `explanation_keywords`

In the offline backend this is token-contribution based. In the transformer path the interface is prepared for attention-style explanations.

## API

### `POST /analyze/text`

Request:

```json
{
  "text": "Your account has been suspended, verify immediately"
}
```

### `POST /analyze/url`

Request:

```json
{
  "url": "http://paypaI-secure-login.com/update"
}
```

### `POST /analyze/full`

Request:

```json
{
  "text": "Your account has been suspended, verify immediately",
  "url": "http://paypaI-secure-login.com/update",
  "html_content": "<html>...</html>"
}
```

### `POST /feedback`

Supports analyst feedback for continuous learning:

```json
{
  "prediction_id": 7,
  "user_label": 1,
  "notes": "Confirmed phishing by SOC analyst"
}
```

## Training

Base training:

```bash
python train_base_model.py
```

Incremental retraining:

```bash
python retrain_incremental.py
```

## Running the API

When FastAPI is installed, run with your ASGI server of choice, for example:

```bash
uvicorn app.main:app --reload
```

In dependency-restricted environments, the project still provides an offline fallback app object used by the test suite.

## Local Demo Frontend

The project also includes a lightweight local showcase site that uses the held-out test split from the latest base-training run.

Start it with:

```bash
python demo_server.py
```

Then open:

```text
http://127.0.0.1:8080
```

The demo site will:

- load a random sample from `app/models/demo_test_samples.jsonl`
- show the source dataset file and expected label
- run inference on that specific held-out sample
- display whether the model was correct
- show phishing probability, explanation keywords, backend name, and URL risk when a URL exists

## Development Setup

Install dependencies:

```bash
pip install -e .[dev]
```

## Security Controls

- input validation on all public routes
- in-memory sliding-window rate limiting
- safe HTML fetching with timeouts and response-size limits
- no script execution or browser automation during HTML analysis
- lightweight domain caching for low-latency repeated checks
- SQLite prediction and feedback audit trail

## Notes About This Offline Sandbox

The current sandbox does not include FastAPI, PyTorch, Transformers, BeautifulSoup, httpx, pandas, or pytest. To keep the project runnable and testable here, the codebase includes offline-safe fallbacks while still exposing the intended production interfaces. In a full environment, installing the dependencies from `pyproject.toml` enables the transformer and FastAPI paths.
