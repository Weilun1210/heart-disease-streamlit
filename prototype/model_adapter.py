"""Model-loading and scoring helpers for the Streamlit research prototype."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURE_ORDER = [
    "Age",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level",
    "Exercise Habits",
    "Smoking",
    "Family Heart Disease",
    "Diabetes",
    "BMI",
    "High Blood Pressure",
    "Low HDL Cholesterol",
    "High LDL Cholesterol",
    "Alcohol Consumption",
    "Stress Level",
    "Sleep Hours",
    "Sugar Consumption",
    "Triglyceride Level",
    "Fasting Blood Sugar",
    "CRP Level",
    "Homocysteine Level",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_path() -> Path:
    override = os.getenv("HEART_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "models" / "notebook_review" / "best_model.joblib"


def resolve_metadata_path(model_path: Path) -> Path | None:
    override = os.getenv("HEART_METADATA_PATH")
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if candidate.exists() else None

    candidates = [
        model_path.with_name("model_metadata.json"),
        model_path.with_name("training_metadata.json"),
        model_path.with_name("metadata.json"),
        model_path.with_name("best_model_metadata.json"),
        project_root() / "models" / "metadata.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def unwrap_model(artifact: Any) -> tuple[Any, dict[str, Any]]:
    """Support either a plain estimator/pipeline or a dictionary wrapper."""
    if not isinstance(artifact, dict):
        return artifact, {}

    for key in ("model", "estimator", "pipeline", "best_model"):
        if key in artifact:
            wrapper_meta = {k: v for k, v in artifact.items() if k != key}
            return artifact[key], wrapper_meta
    raise TypeError("The model artifact is a dictionary without a recognised model key.")


def load_artifacts() -> tuple[Any, dict[str, Any], Path]:
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    model, wrapper_meta = unwrap_model(artifact)
    file_meta = load_metadata(resolve_metadata_path(model_path))
    metadata = {**wrapper_meta, **file_meta}
    return model, metadata, model_path


def decision_threshold(metadata: dict[str, Any]) -> float:
    best_model = metadata.get("best_model")
    nested_threshold = best_model.get("decision_threshold") if isinstance(best_model, dict) else None
    candidates = (
        nested_threshold,
        metadata.get("decision_threshold"),
        metadata.get("threshold"),
        metadata.get("classification_threshold"),
    )
    for value in candidates:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < threshold < 1.0:
            return threshold
    return 0.50


def positive_class_index(model: Any, class_count: int) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        classes = getattr(final, "classes_", None)
    if classes is None:
        return class_count - 1

    normalised = [str(item).strip().lower() for item in list(classes)]
    for label in ("yes", "1", "true", "positive", "heart disease"):
        if label in normalised:
            return normalised.index(label)
    return class_count - 1


def score_one(model: Any, values: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    row = pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(row), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("predict_proba returned an unexpected shape.")
        index = positive_class_index(model, probabilities.shape[1])
        probability = float(probabilities[0, index])
    elif hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(row))[0])
        probability = float(1.0 / (1.0 + np.exp(-score)))
    else:
        prediction = model.predict(row)[0]
        probability = 1.0 if str(prediction).strip().lower() in {"yes", "1", "true"} else 0.0

    if not np.isfinite(probability):
        raise ValueError("The model returned a non-finite probability.")
    return min(max(probability, 0.0), 1.0), row
