# Project Explanation

This document explains how the phishing detection project works end to end, with a focus on how each risk signal is calculated and combined.

## 1. High-Level Architecture

The project analyzes phishing risk from three main angles:

1. **Text / NLP risk**
2. **URL risk**
3. **Structural webpage risk**

These three signals are combined into one final phishing decision:

- `ALLOW`
- `BLOCK`

Core implementation areas:

- Text/model training: [app/ml/training](C:/Users/Totem/Downloads/archive/app/ml/training)
- Synthetic data: [app/ml/synthetic](C:/Users/Totem/Downloads/archive/app/ml/synthetic)
- Retraining: [app/ml/retraining](C:/Users/Totem/Downloads/archive/app/ml/retraining)
- URL analysis: [app/security/url_analyzer.py](C:/Users/Totem/Downloads/archive/app/security/url_analyzer.py)
- Structural analysis: [app/security/web_analyzer.py](C:/Users/Totem/Downloads/archive/app/security/web_analyzer.py)
- Score fusion: [app/services/scoring_service.py](C:/Users/Totem/Downloads/archive/app/services/scoring_service.py)
- Demo server: [demo_server.py](C:/Users/Totem/Downloads/archive/demo_server.py)

## 2. Dataset Pipeline

The dataset loader lives in [app/ml/training/load_local_datasets.py](C:/Users/Totem/Downloads/archive/app/ml/training/load_local_datasets.py).

It does the following:

- scans all CSV files in [dataset](C:/Users/Totem/Downloads/archive/dataset)
- detects text columns from names like `text`, `body`, `message`, `content`, `email`, `text_combined`
- normalizes labels into:
  - `1 = phishing`
  - `0 = benign`
- cleans the text
- extracts URLs from the message body
- removes duplicates
- assigns each record:
  - a `fingerprint`
  - a `group_id`
  - cached `token_counts`

### Why fingerprints and groups exist

These were added to reduce data leakage and overly optimistic evaluation.

- **Fingerprint**
  - derived from a canonicalized version of the email text
  - used to detect near-duplicate messages

- **Group ID**
  - built from source file, sender domain, subject pattern, and body signature
  - used to keep related phishing templates in the same train/validation/test split

This makes the evaluation harder and more realistic.

## 3. Training Model

The training pipeline is in [app/ml/training/train_base_model.py](C:/Users/Totem/Downloads/archive/app/ml/training/train_base_model.py).

### Current active model

The project currently uses:

- **TokenNaiveBayesClassifier**

Implemented in:

- [app/ml/training/model_backends.py](C:/Users/Totem/Downloads/archive/app/ml/training/model_backends.py)

### Intended transformer path

The codebase also includes a transformer backend:

- `TransformerPhishingClassifier`

This is designed for:

- HuggingFace Transformers
- PyTorch
- DistilBERT-style fine-tuning

But the currently trained local model is the Naive Bayes backend.

## 4. How the NLP / Text Risk Works

Text risk is produced by the model service:

- [app/services/model_service.py](C:/Users/Totem/Downloads/archive/app/services/model_service.py)

### Current Naive Bayes approach

The model is a token-frequency classifier:

1. tokenize the message
2. count token occurrences
3. estimate whether tokens are more associated with phishing or benign mail
4. convert the final score into a phishing probability

### What the model learns

The Naive Bayes model keeps:

- phishing token counts
- benign token counts
- vocabulary size
- phishing/benign document counts

For each token it computes a score based on:

- how often it appears in phishing messages
- how often it appears in benign messages

Then it combines all token contributions to produce:

- `phishing_probability`

### Explanation keywords

The NLP explanation output is based on:

- positively weighted phishing tokens
- suspicious keyword hits if no strong token signal is available

Examples of suspicious terms:

- `verify`
- `password`
- `account`
- `security`
- `login`
- `suspended`

## 5. What URL Risk Means

URL risk is calculated in:

- [app/security/url_analyzer.py](C:/Users/Totem/Downloads/archive/app/security/url_analyzer.py)

URL risk answers this question:

**Does the URL itself look like something a phishing attacker would use?**

### URL features checked

The URL analyzer checks:

- **Homoglyph / brand impersonation**
  - example: `paypaI` using uppercase `I` instead of lowercase `l`
  - example: `micros0ft` using zero instead of `o`

- **Suspicious TLDs**
  - such as `.zip`, `.click`, `.top`, `.work`, `.support`

- **Excessive subdomains**
  - phishing URLs often use long chains like:
    - `login.account.verify.portal.example.com`

- **Encoded query strings**
  - long base64-like parameters are suspicious

- **Deep paths**
  - many path segments can indicate an obfuscated phishing landing page

- **Host risk terms**
  - words like:
    - `login`
    - `secure`
    - `verify`
    - `update`
    - `account`

### How URL risk is calculated

The analyzer assigns weighted contributions:

- homoglyph brand impersonation adds a large amount
- suspicious TLD adds risk
- excessive subdomains add risk
- encoded query strings add risk
- deep path adds some risk
- host risk terms add additional risk

The score is clipped to a maximum of `1.0`.

### Example

This URL would score high:

```text
http://paypaI-secure-login.com/update
```

Because it contains:

- brand impersonation (`paypaI`)
- `secure`
- `login`
- phishing-style path

## 6. What Structural Risk Means

Structural risk is calculated in:

- [app/security/web_analyzer.py](C:/Users/Totem/Downloads/archive/app/security/web_analyzer.py)

Structural risk answers this question:

**Does the HTML structure of the page look like a phishing page?**

It does not focus on the URL string. It focuses on page layout and HTML features.

### Structural features checked

The analyzer looks for:

- **Form count**
  - phishing pages often contain credential or payment forms

- **Password fields**
  - a password field on a suspicious page is a strong risk signal

- **Iframe count**
  - iframes can be used to hide or embed content

- **External script count**
  - especially scripts from other domains

- **Brand keyword presence**
  - words like:
    - `paypal`
    - `microsoft`
    - `google`
    - `amazon`
    - `docusign`
    - `okta`

- **Inline obfuscation patterns**
  - things like:
    - `atob(...)`
    - `fromCharCode(...)`

### How structural risk is calculated

The analyzer assigns weighted risk based on:

- number of forms
- presence of password fields
- number of iframes
- number of external scripts
- brand keyword hits
- inline obfuscation hits

The final structural score is also clipped to `1.0`.

### Example

A fake login page like this will raise structural risk:

```html
<html>
  <body>
    <h1>Microsoft 365 Security Verification</h1>
    <form>
      <input type="email">
      <input type="password">
      <button>Verify</button>
    </form>
  </body>
</html>
```

Why:

- contains a form
- contains a password field
- contains a brand-like security/login framing

## 7. How Final Risk Score Is Calculated

Final risk is combined in:

- [app/services/scoring_service.py](C:/Users/Totem/Downloads/archive/app/services/scoring_service.py)

### Inputs

The scoring engine combines:

- `nlp_score`
- `url_score`
- `structural_score`

### Weighted combination

The current weighted score is:

- `45%` NLP
- `35%` URL
- `20%` structural

So the formula is roughly:

```text
final = 0.45 * nlp + 0.35 * url + 0.20 * structural
```

### Extra escalation rules

On top of the weighted sum, the system has rule-based escalation for clearly dangerous combinations, such as:

- strong URL risk + strong NLP risk
- very high NLP risk + login-like structure
- medium URL risk + suspicious structure

### Decision rule

The final decision becomes:

- `BLOCK` if the page passes the configured block conditions
- `ALLOW` otherwise

This is why a page can be blocked even when one single signal is only moderate, if several signals together look phishing-like.

## 8. Leakage and Overfitting Controls

These controls were added to make evaluation more trustworthy.

### Leakage controls

Saved in training metrics under:

- `leakage_checks`

They include:

- label distribution across splits
- group counts per split
- group overlap across train/validation/test
- fingerprint overlap across train/validation/test

The goal is:

- no duplicate or near-duplicate templates crossing the split boundary

### Overfitting controls

Saved in training metrics under:

- `overfitting_checks`

They compare:

- training F1 vs validation F1
- training F1 vs test F1

If the gaps are too large, the run is flagged as:

- `possible_overfitting = true`

## 9. Validation and Test Error Reports

The training pipeline now also saves:

- confusion matrices
- false-positive examples
- false-negative examples
- per-source-file accuracy/error stats

These appear in:

- `validation_error_report`
- `test_error_report`

This makes it easier to explain:

- which phishing patterns are still missed
- which benign messages are being over-flagged

## 10. Synthetic Data and Retraining

Synthetic phishing generation is in:

- [app/ml/synthetic/generator.py](C:/Users/Totem/Downloads/archive/app/ml/synthetic/generator.py)

It creates:

- phishing-like enterprise login prompts
- urgent account verification messages
- homoglyph phishing URLs
- encoded query strings

Incremental retraining is in:

- [app/ml/retraining/retrain_incremental.py](C:/Users/Totem/Downloads/archive/app/ml/retraining/retrain_incremental.py)

It:

- loads the saved model
- loads synthetic phishing data
- loads analyst feedback from SQLite
- updates the model
- compares before/after metrics

## 11. Browser Extension and Demo

The browser extension is in:

- [browser_extension](C:/Users/Totem/Downloads/archive/browser_extension)

The local demo server is:

- [demo_server.py](C:/Users/Totem/Downloads/archive/demo_server.py)

The extension:

- reads the active tab
- sends page URL + text + HTML snapshot to the local detector
- receives:
  - decision
  - final score
  - URL risk
  - structural risk
  - NLP score
  - explanation keywords

### Email Mode

The extension also has an email-specific mode for:

- Gmail
- Outlook Web

When a supported webmail page is detected, the popup tries to extract:

- sender
- subject
- visible message body
- message links from the email body itself

This is important because the generic webpage pipeline would otherwise treat the Gmail or Outlook shell as if it were the suspicious page, which can create misleading structural signals.

### How email scoring differs from webpage scoring

Email mode is handled in:

- [demo_server.py](C:/Users/Totem/Downloads/archive/demo_server.py)

through the `POST /api/analyze/email` path.

That flow:

- builds a combined text signal from subject + body
- runs the NLP model on the email text
- analyzes embedded links separately through the URL engine
- does not use webpage structural risk from Gmail or Outlook chrome
- uses mailbox context when available, such as a Gmail spam-folder location

### Email risk calculation

The current email score is a weighted blend of:

- `75%` NLP phishing probability
- `25%` maximum embedded-link URL risk

Then the score is adjusted with email-specific heuristics:

- reduce risk for trusted sender domains with low-risk links
- reduce risk when links are safe and there are few credential-oriented keywords
- increase risk when the mailbox context indicates the spam folder

The final email decision also includes rule-based blocking for combinations such as:

- very risky embedded links
- very high NLP phishing probability plus elevated link risk
- credential-heavy text plus non-trivial link risk
- very high NLP phishing probability from a non-trusted sender while the message is already in spam

That design is intentional: email mode is meant to behave differently from webpage mode because the signals are different.

### Why email links can be imperfect in webmail

Webmail clients sometimes rewrite, hide, or virtualize the real message links. In Gmail especially, the DOM can expose:

- Gmail navigation links
- support links
- mailbox chrome links

instead of the real campaign links.

The extension filters common Gmail and Outlook chrome links, but real-message extraction can still be imperfect depending on the mailbox layout and DOM state. That is why the email mode also uses:

- sender context
- message text
- spam-folder hint

instead of depending only on extracted links.

### Operational note

After changing:

- [demo_server.py](C:/Users/Totem/Downloads/archive/demo_server.py)
- [browser_extension/popup.js](C:/Users/Totem/Downloads/archive/browser_extension/popup.js)

you should:

1. restart `python demo_server.py`
2. reload the unpacked extension in Chrome or Edge

Otherwise the popup may still be talking to an older local server process or older extension bundle.

## 12. What Is Still Missing

The project is production-style and functional, but a few original ambitions are still only partially implemented:

- active DistilBERT fine-tuning as the real trained model
- true SHAP or transformer attention explainability
- full dependency-backed FastAPI runtime in active use
- full pytest execution in every environment by default

## 13. Short Summary

In simple terms:

- **URL risk** asks whether the link itself looks suspicious
- **Structural risk** asks whether the page layout/HTML looks like phishing
- **NLP risk** asks whether the text sounds like phishing
- **Final risk score** combines all of them into one decision

That is the core logic behind the system.
