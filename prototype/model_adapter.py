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

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_folder() -> Path:
    return project_root() / "models" / "notebook_review"


def resolve_model_path() -> Path:
    override = os.getenv("HEART_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return model_folder() / "best_model.joblib"


def resolve_metadata_path(model_path: Path | None = None) -> Path | None:
    override = os.getenv("HEART_METADATA_PATH")
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if candidate.exists() else None

    candidates = []
    if model_path is not None:
        candidates.extend(
            [
                model_path.with_name("model_metadata.json"),
                model_path.with_name("training_metadata.json"),
                model_path.with_name("metadata.json"),
                model_path.with_name("best_model_metadata.json"),
            ]
        )
    candidates.extend(
        [
            model_folder() / "model_metadata.json",
            project_root() / "models" / "metadata.json",
        ]
    )
    return next((path for path in candidates if path.exists()), None)


def load_metadata(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = resolve_metadata_path()
    if path is None or not path.exists():
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


def load_model_collection() -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Load every per-model artifact that exists, with best_model as a safe fallback."""
    models: dict[str, Any] = {}
    paths: dict[str, Path] = {}

    folder = model_folder()
    for display_name, filename in MODEL_FILES.items():
        path = folder / filename
        if path.exists():
            artifact = joblib.load(path)
            model, _ = unwrap_model(artifact)
            models[display_name] = model
            paths[display_name] = path

    metadata = load_metadata(resolve_metadata_path())

    if not models:
        best_model, metadata, best_path = load_artifacts()
        best_meta = metadata.get("best_model") if isinstance(metadata.get("best_model"), dict) else {}
        best_name = str(
            best_meta.get("display_name")
            or best_meta.get("name")
            or metadata.get("model_name")
            or "Random Forest"
        )
        models[best_name] = best_model
        paths[best_name] = best_path

    return models, paths, metadata


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


def build_row(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def score_model(model: Any, values: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score one row without pretending an SVM margin is a calibrated probability."""
    row = build_row(values)
    prediction = model.predict(row)[0]
    positive = str(prediction).strip().lower() in {"yes", "1", "true"}

    result: dict[str, Any] = {
        "prediction": "Yes" if positive else "No",
        "positive": bool(positive),
        "score_kind": "class only",
        "display_score": None,
        "raw_score": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(row), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("predict_proba returned an unexpected shape.")
        index = positive_class_index(model, probabilities.shape[1])
        probability = float(probabilities[0, index])
        if not np.isfinite(probability):
            raise ValueError("The model returned a non-finite probability.")
        result.update(
            {
                "score_kind": "positive-class probability",
                "display_score": min(max(probability, 0.0), 1.0),
                "raw_score": probability,
            }
        )
    elif hasattr(model, "decision_function"):
        margin = float(np.ravel(model.decision_function(row))[0])
        if not np.isfinite(margin):
            raise ValueError("The model returned a non-finite decision score.")
        result.update(
            {
                "score_kind": "decision-function margin",
                "display_score": margin,
                "raw_score": margin,
            }
        )

    return result, row


def score_one(model: Any, values: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    """Backward-compatible helper used by the original app."""
    result, row = score_model(model, values)
    if result["score_kind"] == "positive-class probability":
        return float(result["display_score"]), row
    if result["score_kind"] == "decision-function margin":
        margin = float(result["display_score"])
        return float(1.0 / (1.0 + np.exp(-margin))), row
    return (1.0 if result["positive"] else 0.0), row
