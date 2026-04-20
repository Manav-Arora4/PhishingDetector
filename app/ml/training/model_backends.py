"""Training backends for phishing detection."""

from __future__ import annotations

import json
import math
import pickle
import random
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence

from app.utils.config import Settings
from app.utils.text import clean_text, suspicious_keyword_hits, tokenize

try:  # pragma: no cover - optional optimization path.
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numpy is unavailable.
    np = None
    NUMPY_AVAILABLE = False


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_from_predictions(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    """Compute common binary classification metrics without third-party dependencies."""

    if not labels:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    true_positive = sum(1 for truth, pred in zip(labels, predictions, strict=False) if truth == 1 and pred == 1)
    true_negative = sum(1 for truth, pred in zip(labels, predictions, strict=False) if truth == 0 and pred == 0)
    false_positive = sum(1 for truth, pred in zip(labels, predictions, strict=False) if truth == 0 and pred == 1)
    false_negative = sum(1 for truth, pred in zip(labels, predictions, strict=False) if truth == 1 and pred == 0)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(true_positive + true_negative, len(labels))
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        },
    }


def train_validation_test_split(
    samples: Sequence[tuple[str, int]],
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a deterministic train/validation/test split."""

    materialized = list(samples)
    if len(materialized) < 3:
        return materialized, materialized, materialized
    total_ratio = train_ratio + validation_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1.0.")
    random.Random(seed).shuffle(materialized)
    train_end = max(1, int(len(materialized) * train_ratio))
    validation_end = train_end + max(1, int(len(materialized) * validation_ratio))
    train_end = min(train_end, len(materialized) - 2)
    validation_end = min(max(validation_end, train_end + 1), len(materialized) - 1)
    train_samples = materialized[:train_end]
    validation_samples = materialized[train_end:validation_end]
    test_samples = materialized[validation_end:]
    if not test_samples:
        test_samples = validation_samples[-1:]
        validation_samples = validation_samples[:-1] or train_samples[-1:]
    return train_samples, validation_samples, test_samples


def grouped_train_validation_test_split(
    samples: Sequence[tuple[Any, Hashable]],
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split samples by group id to keep related templates in the same split."""

    materialized = list(samples)
    if len(materialized) < 3:
        payload = [sample for sample, _group in materialized]
        return payload, payload, payload
    total_ratio = train_ratio + validation_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must sum to 1.0.")

    grouped: dict[Hashable, list[Any]] = {}
    for sample, group in materialized:
        grouped.setdefault(group, []).append(sample)

    ordered_groups = list(grouped.items())
    random.Random(seed).shuffle(ordered_groups)
    total_count = sum(len(group_samples) for _group, group_samples in ordered_groups)
    train_target = total_count * train_ratio
    validation_target = total_count * validation_ratio

    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    test_samples: list[Any] = []

    for _group, group_samples in ordered_groups:
        if len(train_samples) < train_target:
            train_samples.extend(group_samples)
        elif len(validation_samples) < validation_target:
            validation_samples.extend(group_samples)
        else:
            test_samples.extend(group_samples)

    if not validation_samples and test_samples:
        validation_samples.extend(test_samples[:1])
        test_samples = test_samples[1:]
    if not test_samples and validation_samples:
        test_samples.extend(validation_samples[-1:])
        validation_samples = validation_samples[:-1] or train_samples[-1:]
    if not train_samples:
        train_samples.extend(validation_samples[:1] or test_samples[:1])
    return train_samples, validation_samples, test_samples


class BasePhishingModel(ABC):
    """Abstract phishing classification backend."""

    backend_name = "base"

    @abstractmethod
    def fit(self, samples: Sequence[tuple[str, int]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, samples: Sequence[tuple[str, int]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def explain(self, text: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BasePhishingModel":
        raise NotImplementedError

    @abstractmethod
    def snapshot_weights(self) -> dict[str, Any]:
        raise NotImplementedError


class TokenNaiveBayesClassifier(BasePhishingModel):
    """Offline-safe multinomial Naive Bayes model with interpretable token weights."""

    backend_name = "token_naive_bayes"

    def __init__(self) -> None:
        self.phishing_counts: Counter[str] = Counter()
        self.benign_counts: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.phishing_docs = 0
        self.benign_docs = 0
        self._token_score_cache: dict[str, float] = {}
        self._phishing_prior = 0.0
        self._benign_prior = 0.0
        self._token_index: dict[str, int] = {}
        self._token_weights = None

    def fit(self, samples: Sequence[tuple[str, int]]) -> None:
        self.phishing_counts.clear()
        self.benign_counts.clear()
        self.vocabulary.clear()
        self.phishing_docs = 0
        self.benign_docs = 0
        self.partial_fit(samples)

    def partial_fit(self, samples: Sequence[tuple[str, int]]) -> None:
        for text, label in samples:
            tokens = tokenize(text)
            if not tokens:
                continue
            self.vocabulary.update(tokens)
            if label == 1:
                self.phishing_docs += 1
                self.phishing_counts.update(tokens)
            else:
                self.benign_docs += 1
                self.benign_counts.update(tokens)
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        total_docs = max(self.phishing_docs + self.benign_docs, 1)
        self._phishing_prior = math.log((self.phishing_docs + 1) / (total_docs + 2))
        self._benign_prior = math.log((self.benign_docs + 1) / (total_docs + 2))
        ordered_tokens = sorted(self.vocabulary)
        self._token_score_cache = {token: self._token_score(token) for token in ordered_tokens}
        self._token_index = {token: index for index, token in enumerate(ordered_tokens)}
        if NUMPY_AVAILABLE:
            self._token_weights = np.array(
                [self._token_score_cache[token] for token in ordered_tokens],
                dtype=np.float32,
            )
        else:
            self._token_weights = None

    def _token_score(self, token: str) -> float:
        vocab_size = max(len(self.vocabulary), 1)
        phishing_total = sum(self.phishing_counts.values()) + vocab_size
        benign_total = sum(self.benign_counts.values()) + vocab_size
        phishing_likelihood = (self.phishing_counts[token] + 1) / phishing_total
        benign_likelihood = (self.benign_counts[token] + 1) / benign_total
        return math.log(phishing_likelihood) - math.log(benign_likelihood)

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        token_count_batches = [Counter(tokenize(text)) for text in texts]
        keyword_hits = [len(suspicious_keyword_hits(text)) for text in texts]
        return self.predict_proba_from_token_counts(token_count_batches, keyword_hit_counts=keyword_hits)

    def predict_proba_from_token_counts(
        self,
        token_count_maps: Sequence[dict[str, int] | Counter[str]],
        *,
        keyword_hit_counts: Sequence[int] | None = None,
    ) -> list[float]:
        """Score pre-tokenized samples for faster repeated evaluation."""

        scores: list[float] = []
        for index, token_counts in enumerate(token_count_maps):
            delta_sum = self._score_token_counts(token_counts)
            phishing_log_prob = self._phishing_prior + max(delta_sum, 0.0)
            benign_log_prob = self._benign_prior + max(-delta_sum, 0.0)
            suspicious_bonus = 0.18 * (keyword_hit_counts[index] if keyword_hit_counts is not None else 0)
            logit = (phishing_log_prob - benign_log_prob) + suspicious_bonus
            scores.append(round(1.0 / (1.0 + math.exp(-max(min(logit, 20.0), -20.0))), 4))
        return scores

    def _score_token_counts(self, token_counts: dict[str, int] | Counter[str]) -> float:
        if NUMPY_AVAILABLE and self._token_weights is not None and self._token_index:
            indices: list[int] = []
            values: list[int] = []
            fallback_sum = 0.0
            for token, count in token_counts.items():
                token_index = self._token_index.get(token)
                if token_index is None:
                    delta = self._token_score_cache.get(token)
                    if delta is None:
                        delta = self._token_score(token)
                        self._token_score_cache[token] = delta
                    fallback_sum += delta * count
                else:
                    indices.append(token_index)
                    values.append(count)
            if indices:
                value_array = np.array(values, dtype=np.float32)
                weight_array = self._token_weights[np.array(indices, dtype=np.int32)]
                return float(weight_array.dot(value_array)) + fallback_sum
            return fallback_sum

        delta_sum = 0.0
        for token, count in token_counts.items():
            delta = self._token_score_cache.get(token)
            if delta is None:
                delta = self._token_score(token)
                self._token_score_cache[token] = delta
            delta_sum += delta * count
        return delta_sum

    def explain(self, text: str, top_k: int = 5) -> list[str]:
        token_scores = {token: self._token_score(token) for token in tokenize(text)}
        ranked = [token for token, score in sorted(token_scores.items(), key=lambda item: item[1], reverse=True) if score > 0]
        if not ranked:
            return suspicious_keyword_hits(text)[:top_k]
        return ranked[:top_k]

    def save(self, path: Path) -> None:
        payload = {
            "backend": self.backend_name,
            "phishing_counts": dict(self.phishing_counts),
            "benign_counts": dict(self.benign_counts),
            "vocabulary": sorted(self.vocabulary),
            "phishing_docs": self.phishing_docs,
            "benign_docs": self.benign_docs,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "TokenNaiveBayesClassifier":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls()
        instance.phishing_counts.update(payload.get("phishing_counts", {}))
        instance.benign_counts.update(payload.get("benign_counts", {}))
        instance.vocabulary.update(payload.get("vocabulary", []))
        instance.phishing_docs = int(payload.get("phishing_docs", 0))
        instance.benign_docs = int(payload.get("benign_docs", 0))
        instance._refresh_cache()
        return instance

    @classmethod
    def bootstrap_default(cls) -> "TokenNaiveBayesClassifier":
        model = cls()
        seed_samples = [
            ("Your account has been suspended. Verify immediately to avoid closure.", 1),
            ("Payroll update required. Confirm your password using the secure portal.", 1),
            ("Weekly team sync moved to 3 PM. Agenda attached.", 0),
            ("Lunch order has been placed and your receipt is available.", 0),
            ("Can we reschedule the project review to Friday morning?", 0),
            ("Please review the meeting agenda before tomorrow's planning session.", 0),
        ]
        model.fit(seed_samples)
        return model

    def snapshot_weights(self) -> dict[str, Any]:
        return {
            "phishing_docs": self.phishing_docs,
            "benign_docs": self.benign_docs,
            "phishing_token_total": sum(self.phishing_counts.values()),
            "benign_token_total": sum(self.benign_counts.values()),
            "vocabulary_size": len(self.vocabulary),
        }


class TransformerPhishingClassifier(BasePhishingModel):
    """Optional HuggingFace/PyTorch backend used when dependencies are present."""

    backend_name = "transformer"

    def __init__(self, model_name: str, allow_download: bool = False) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Transformers backend requires torch and transformers.") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            local_files_only=not allow_download,
        )
        self.model_name = model_name

    def fit(self, samples: Sequence[tuple[str, int]]) -> None:
        self._run_training(samples, epochs=1)

    def partial_fit(self, samples: Sequence[tuple[str, int]]) -> None:
        self._run_training(samples, epochs=1)

    def _run_training(self, samples: Sequence[tuple[str, int]], epochs: int) -> None:
        if not samples:
            return
        encoded = self._tokenizer(
            [clean_text(text, max_length=512) for text, _ in samples],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self._torch.tensor([label for _, label in samples])
        optimizer = self._torch.optim.AdamW(self._model.parameters(), lr=2e-5)
        self._model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self._model(**encoded, labels=labels)
            outputs.loss.backward()
            optimizer.step()

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        encoded = self._tokenizer(
            [clean_text(text, max_length=512) for text in texts],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self._model.eval()
        with self._torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=1)[:, 1].tolist()
        return [round(float(value), 4) for value in probabilities]

    def explain(self, text: str, top_k: int = 5) -> list[str]:
        return suspicious_keyword_hits(text)[:top_k]

    def save(self, path: Path) -> None:
        payload = {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "state_dict": self._model.state_dict(),
        }
        self._torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "TransformerPhishingClassifier":
        import torch  # pragma: no cover

        payload = torch.load(path, map_location="cpu")
        instance = cls(model_name=payload["model_name"])
        instance._model.load_state_dict(payload["state_dict"])
        return instance

    def snapshot_weights(self) -> dict[str, Any]:
        state_dict = self._model.state_dict()
        return {name: float(tensor.float().abs().sum().item()) for name, tensor in list(state_dict.items())[:4]}


def select_training_backend(settings: Settings | None = None, force_backend: str | None = None) -> BasePhishingModel:
    """Select the best available backend for the current runtime."""

    effective = settings or Settings()
    if force_backend == TokenNaiveBayesClassifier.backend_name:
        return TokenNaiveBayesClassifier()
    if force_backend == TransformerPhishingClassifier.backend_name:
        return TransformerPhishingClassifier(
            model_name=effective.default_model_name,
            allow_download=effective.allow_model_download,
        )
    try:
        return TransformerPhishingClassifier(
            model_name=effective.default_model_name,
            allow_download=effective.allow_model_download,
        )
    except Exception:
        return TokenNaiveBayesClassifier()


def load_saved_model(path: Path) -> BasePhishingModel:
    """Load a saved phishing model artifact."""

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        backend = payload.get("backend")
        if backend == TokenNaiveBayesClassifier.backend_name:
            return TokenNaiveBayesClassifier.load(path)
    except Exception:
        pass
    try:
        return TransformerPhishingClassifier.load(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load model from {path}.") from exc


def load_or_bootstrap_model(path: Path | None = None, settings: Settings | None = None) -> BasePhishingModel:
    """Load the persisted model or provide a small bootstrapped fallback model."""

    effective = settings or Settings()
    target = path or effective.model_artifact_path
    if target.exists():
        try:
            return load_saved_model(target)
        except Exception:
            return TokenNaiveBayesClassifier.bootstrap_default()
    return TokenNaiveBayesClassifier.bootstrap_default()


def evaluate_model(
    model: BasePhishingModel,
    samples: Sequence[tuple[str, int]],
    *,
    progress_callback: Callable[[str], None] | None = None,
    stage_name: str = "evaluation",
    batch_size: int = 64,
    token_counts_sequence: Sequence[dict[str, int]] | None = None,
    keyword_hit_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run evaluation and package both metrics and example probabilities."""

    labels = [label for _, label in samples]
    probabilities: list[float] = []
    texts = [text for text, _ in samples]
    if texts:
        total_batches = max(1, math.ceil(len(texts) / batch_size))
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            stop = start + batch_size
            batch = texts[start:stop]
            if progress_callback:
                progress_callback(
                    f"{stage_name}: starting batch {batch_index}/{total_batches} ({start + 1}-{min(stop, len(texts))} of {len(texts)})"
                )
            if (
                token_counts_sequence is not None
                and hasattr(model, "predict_proba_from_token_counts")
            ):
                batch_token_counts = token_counts_sequence[start:stop]
                batch_keyword_hits = keyword_hit_counts[start:stop] if keyword_hit_counts is not None else None
                probabilities.extend(
                    model.predict_proba_from_token_counts(
                        batch_token_counts,
                        keyword_hit_counts=batch_keyword_hits,
                    )
                )
            else:
                probabilities.extend(model.predict_proba(batch))
            if progress_callback:
                progress_callback(
                    f"{stage_name}: processed batch {batch_index}/{total_batches} ({min(start + len(batch), len(texts))}/{len(texts)} samples)"
                )
    metrics = metrics_from_predictions(labels, probabilities)
    metrics["sample_count"] = len(samples)
    metrics["mean_probability"] = round(sum(probabilities) / len(probabilities), 4) if probabilities else 0.0
    return metrics


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
