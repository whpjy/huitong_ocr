"""Optional four-way document orientation classification."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

_MODEL_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_ERROR: str | None = None
_SETTINGS: tuple[bool, str, str] | None = None


def _settings() -> tuple[bool, str, str]:
    global _SETTINGS
    if _SETTINGS is None:
        try:
            from .config import load_config

            item = load_config().orientation
            _SETTINGS = (item.enabled, item.model, item.device)
        except Exception:
            _SETTINGS = (True, "PP-LCNet_x1_0_doc_ori", "cpu")
    enabled, model, device = _SETTINGS
    env_enabled = os.getenv("OCR_ORIENTATION_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() not in {"0", "false", "no", "off"}
    return (
        enabled,
        os.getenv("OCR_ORIENTATION_MODEL", model),
        os.getenv("OCR_ORIENTATION_DEVICE", device),
    )


def _enabled() -> bool:
    return _settings()[0]


def _model() -> Any:
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None or _MODEL_ERROR is not None:
            return _MODEL
        try:
            from paddleocr import DocImgOrientationClassification

            _MODEL = DocImgOrientationClassification(
                model_name=_settings()[1],
                device=_settings()[2],
            )
        except Exception as exc:  # Optional runtime dependency.
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
    return _MODEL


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    for attr in ("json", "to_json"):
        candidate = getattr(value, attr, None)
        if callable(candidate):
            try:
                parsed = candidate()
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        elif isinstance(candidate, dict):
            return candidate
    return None


def _find_angle(value: Any) -> tuple[int, float | None] | None:
    mapping = _as_mapping(value)
    if mapping is not None:
        score = mapping.get("scores")
        if isinstance(score, (list, tuple)) and score:
            score = score[0]
        confidence = float(score) if isinstance(score, (int, float)) else None
        for key in ("angle", "label_names", "class_ids"):
            item = mapping.get(key)
            if key == "angle" and isinstance(item, (int, float)):
                angle = int(item) % 360
                return angle, confidence
            if key == "class_ids" and isinstance(item, (list, tuple)) and item:
                item = item[0]
            if key == "label_names" and isinstance(item, (list, tuple)) and item:
                item = item[0]
            if key == "class_ids" and isinstance(item, (int, float)):
                return (int(item) % 4) * 90, confidence
            if key == "label_names" and isinstance(item, str):
                digits = "".join(ch for ch in item if ch.isdigit())
                if digits:
                    return int(digits) % 360, confidence
        for nested in mapping.values():
            found = _find_angle(nested)
            if found:
                return found[0], confidence if confidence is not None else found[1]
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_angle(item)
            if found:
                return found
    return None


def detect_document_orientation(path: Path) -> dict[str, Any]:
    """Return the predicted source angle; never raises for optional failures."""

    return detect_document_orientations([path])[0]


def detect_document_orientations(paths: list[Path]) -> list[dict[str, Any]]:
    """Classify independent document images in one predictor batch."""

    defaults = [
        {
            "enabled": _enabled(),
            "angle": 0,
            "confidence": None,
            "source": "fallback",
            "error": None,
        }
        for _path in paths
    ]
    if not paths:
        return defaults
    if not _enabled():
        for result in defaults:
            result["source"] = "disabled"
        return defaults
    model = _model()
    if model is None:
        for result in defaults:
            result["error"] = _MODEL_ERROR
        return defaults
    try:
        with _PREDICT_LOCK:
            predictions = list(
                model.predict(
                    [str(path) for path in paths],
                    batch_size=len(paths),
                )
            )
        for index, result in enumerate(defaults):
            if index >= len(predictions):
                result["error"] = "orientation batch result count mismatch"
                continue
            found = _find_angle(predictions[index])
            if found is None:
                result["error"] = "orientation result did not contain an angle"
                continue
            result.update(
                angle=found[0],
                confidence=found[1],
                source="PP-LCNet_x1_0_doc_ori",
            )
    except Exception as exc:  # Orientation must never block OCR.
        error = f"{type(exc).__name__}: {exc}"
        for result in defaults:
            result["error"] = error
    return defaults
