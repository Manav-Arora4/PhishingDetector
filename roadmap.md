# Phishing Detection System – Development Roadmap

## Overview
This roadmap outlines the step-by-step development of a phishing detection system using machine learning and NLP. The project will start with a simple prototype and progressively incorporate more advanced techniques.

---

## Phase 1: Dataset Collection & Analysis

**Goal:** Understand phishing patterns and useful detection features.

### Tasks
- Collect datasets from multiple sources.

**Email/Text**
- Enron Email Dataset
- SpamAssassin Dataset
- Nazario Phishing Corpus

**URL**
- PhishTank
- OpenPhish
- Tranco / Majestic Top Sites

**Webpages**
- PhishTank archived webpages
- Common Crawl
- Kaggle phishing datasets

### Exploratory Data Analysis (EDA)

Analyze patterns in:

**Text**
- Urgency language
- Credential requests
- Suspicious phrases

**URLs**
- URL length
- Number of subdomains
- Presence of IP address
- Suspicious TLDs

**Webpages**
- Login forms
- External scripts
- Hidden elements

Tools:
- Python
- pandas
- matplotlib

---

## Phase 2: Baseline NLP Classifier

**Goal:** Build the first working phishing classifier.

### Steps
1. Convert email/SMS text into features using:
   - TF-IDF
   - Bag-of-Words

2. Train baseline models:
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - SVM

3. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score

---

## Phase 3: URL-Based Phishing Detection

**Goal:** Detect phishing using URL features.

### Example Features

- URL length
- Number of dots
- Number of subdomains
- Presence of `@`
- Presence of IP address
- Redirect count

### Models
- Random Forest
- Gradient Boosting
- XGBoost

---

## Phase 4: Transformer NLP Model

**Goal:** Improve text detection using deep learning.

Models:
- BERT
- DistilBERT
- RoBERTa

Pipeline:

Text → Tokenizer → Transformer → Classifier → Output

Tools:
- HuggingFace Transformers
- PyTorch

---

## Phase 5: Website Structure Analysis

**Goal:** Detect phishing through webpage structure.

Extract HTML features such as:

- Number of forms
- Login inputs
- External scripts
- iframe usage
- Hidden elements

Optional:
- CNN on webpage screenshots for brand impersonation detection.

---

## Phase 6: Real-Time Detection System

**Goal:** Deploy models into a real-time service.

Architecture:

Browser → Feature Extraction → Detection API → ML Model → Risk Score

Backend tools:
- FastAPI / Flask
- Docker

---

## Phase 7: Browser Extension

**Goal:** Provide real-time phishing protection.

Features:
- Monitor visited URLs
- Extract page metadata
- Send data to detection API
- Display phishing warnings

Example:

Warning: This page may be a phishing site.

---

## Phase 8: Continuous Learning

**Goal:** Keep the model updated.

Pipeline:

New Threats → Dataset Update → Model Retraining → Validation → Deployment

Tools:
- Airflow
- MLflow

---

## Suggested Timeline

| Week | Task |
|-----|------|
| 1–2 | Dataset collection + EDA |
| 3 | Baseline NLP model |
| 4 | URL classifier |
| 5–6 | Transformer NLP model |
| 7 | Website analysis |
| 8 | Real-time API |
| 9 | Browser extension |
| 10 | Testing & evaluation |

---

## Minimum Viable System

Text Classifier  
+  
URL Feature Classifier  
↓  
Combined Model  
↓  
Phishing Risk Score