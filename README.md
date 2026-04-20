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

## Dataset Source

The project loads datasets only from the local [`dataset/`](./dataset) directory at runtime.

Reference source for the phishing email corpus:

- [Kaggle: Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)

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

Useful training options:

```bash
python train_base_model.py --max-rows-per-file 1000 --backend token_naive_bayes
python train_base_model.py --max-rows-per-file 1000 --backend token_naive_bayes --rebuild-cache
python train_base_model.py --max-rows-per-file 1000 --backend token_naive_bayes --no-cache
```

The training script now supports:

- preprocessing cache reuse for normalized dataset artifacts
- file-by-file preprocessing progress logs
- cache rebuild control for fresh preprocessing runs
- resume-style repeated runs that skip expensive CSV normalization when the cache is valid

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

## Browser Extension Demo

A local browser extension demo is included under [`browser_extension`](./browser_extension).

It can inspect the active browser tab and send:

- the current page URL
- page text
- page HTML snapshot

to the local phishing detector server for analysis.

Setup:

```bash
python demo_server.py
```

Then in Chrome or Edge:

1. Open the extensions page
2. Enable Developer Mode
3. Choose `Load unpacked`
4. Select the [`browser_extension`](./browser_extension) folder

The extension popup will then analyze the active tab against your local phishing detection stack.

### Email Mode

The extension also supports an email-focused mode for browser-based webmail views.

When the active tab appears to be:

- Gmail
- Outlook Web

the popup will try to extract:

- sender
- subject
- visible message body
- embedded links inside the email

It then sends the email content through the local phishing detector and shows:

- phishing decision
- final risk score
- NLP risk
- maximum embedded-link URL risk
- structural risk as `N/A` because the surrounding Gmail/Outlook shell is not treated as the target phishing page
- explanation keywords
- up to a few analyzed links extracted from the email

Email mode uses a separate endpoint:

- `POST /api/analyze/email`

and applies email-specific logic instead of the generic webpage scoring rule. That email-specific path:

- analyzes the sender, subject, and visible message body
- scores embedded links separately
- avoids using Gmail or Outlook page chrome as structural phishing evidence
- uses mailbox context such as Gmail spam-folder location when available

This helps reduce false positives on legitimate webmail pages and helps catch suspicious messages even when the visible message links are partially hidden by the webmail client.

### Extension Troubleshooting

If the popup shows an outdated result or a JSON parsing error such as `Unexpected token '<'`, the most common cause is a stale local server process or stale unpacked extension bundle.

Refresh both sides:

1. Stop the local server with `Ctrl + C`
2. Restart it with `python demo_server.py`
3. Open `chrome://extensions` or `edge://extensions`
4. Click `Reload` for the unpacked extension
5. Reopen the popup and run the analysis again

This is especially important after changes to:

- `demo_server.py`
- `browser_extension/popup.js`

## Safe URL Testing

You do not need to visit a live phishing site to test the URL analysis engine.

Start the local server:

```bash
python demo_server.py
```

Then test a URL safely through the local API:

```bash
python -c "import requests; print(requests.post('http://127.0.0.1:8080/api/analyze/url', json={'url':'http://paypaI-secure-login.com/update'}).json())"
```

You can also test the URL engine directly without starting the server:

```bash
python -c "import asyncio; from app.security.url_analyzer import URLAnalysisEngine; print(asyncio.run(URLAnalysisEngine().analyze('http://paypaI-secure-login.com/update')).as_dict())"
```

Safe example URLs:

- `http://paypaI-secure-login.com/update`
- `http://login.account.verify.micros0ft-support.top/auth`
- `http://secure.portal.g00gle.work/reset?continue=ZXZlbnQ9dmVyaWZ5`
- `https://example.com/about`

Expected outcome:

- phishing-style URLs should return higher risk scores
- normal URLs such as `https://example.com/about` should return low risk scores

## Safe Email Testing

You do not need to open a live phishing campaign to test email detection.

Safer ways to test email mode:

- open a message in Gmail or Outlook Web and let the extension analyze the visible message body
- use known spam-folder examples in a test mailbox
- send controlled synthetic phishing samples to a disposable inbox
- call the local email endpoint directly with sample sender, subject, body, and links

Example direct email test:

```bash
python -c "import requests; print(requests.post('http://127.0.0.1:8080/api/analyze/email', json={'sender':'alerts@example-security.com','subject':'Verify your account now','body':'Your account has been suspended. Click below to verify immediately.','links':['http://paypaI-secure-login.com/update'],'mailbox_hint':'#spam/test'}).json())"
```

Expected behavior:

- suspicious sender + urgent credential language + risky links should trend toward `BLOCK`
- legitimate newsletters or product emails with safe links should usually trend toward `ALLOW`
- email mode may still assign a moderate score to marketing-style language, but the decision is based on the email-specific heuristics rather than webpage structure

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

The current sandbox may not include the full runtime stack used by the project, including FastAPI, PyTorch, Transformers, or browser/runtime helpers. To keep the project runnable and testable here, the codebase includes offline-safe fallbacks while still exposing the intended production interfaces. In a full environment, installing the dependencies from `pyproject.toml` enables the transformer and FastAPI paths.
