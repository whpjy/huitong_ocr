"""Backend-owned recognition model selection for the mobile client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_MOBILE_MODEL_KEYS = {
    "multimodal:hunyuan_ocr",
    "hybrid:hunyuan_ocr",
}


class MobileConfigError(ValueError):
    """Raised when the mobile recognition configuration is invalid."""


@dataclass(frozen=True)
class MobileRecognitionConfig:
    name: str
    model_key: str
    label: str
    image_quality_enabled: bool = False

    @property
    def pipeline_type(self) -> str:
        return self.model_key.partition(":")[0]


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MobileConfigError(f"{location} 必须是对象")
    return value


def load_mobile_config(
    path: str | Path | None = None,
) -> MobileRecognitionConfig:
    config_path = Path(path) if path else (
        Path(__file__).resolve().parents[1] / "config" / "mobile.yaml"
    )
    if not config_path.is_file():
        raise MobileConfigError(f"移动端配置文件不存在：{config_path}")
    try:
        raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MobileConfigError(f"移动端 YAML 解析失败：{exc}") from exc

    raw = _mapping(raw_value, "root")
    if raw.get("version") != 1:
        raise MobileConfigError("仅支持移动端配置版本 version: 1")
    mobile = _mapping(raw.get("mobile"), "mobile")
    image_quality = _mapping(
        mobile.get("image_quality", {}),
        "mobile.image_quality",
    )
    image_quality_enabled = image_quality.get("enabled", False)
    if not isinstance(image_quality_enabled, bool):
        raise MobileConfigError(
            "mobile.image_quality.enabled 必须是 true 或 false"
        )
    models = _mapping(mobile.get("models"), "mobile.models")
    enabled: list[MobileRecognitionConfig] = []
    for raw_name, raw_model in models.items():
        name = str(raw_name).strip()
        model = _mapping(raw_model, f"mobile.models.{name}")
        flag = model.get("enabled", False)
        if not isinstance(flag, bool):
            raise MobileConfigError(
                f"mobile.models.{name}.enabled 必须是 true 或 false"
            )
        if not flag:
            continue
        model_key = str(model.get("model_key") or "").strip()
        label = str(model.get("label") or "").strip()
        if model_key not in SUPPORTED_MOBILE_MODEL_KEYS:
            raise MobileConfigError(
                f"移动端模型不受支持：{model_key or '(空)'}"
            )
        if not label:
            raise MobileConfigError(f"mobile.models.{name}.label 不能为空")
        enabled.append(
            MobileRecognitionConfig(
                name,
                model_key,
                label,
                image_quality_enabled=image_quality_enabled,
            )
        )
    if len(enabled) != 1:
        raise MobileConfigError(
            "mobile.models 必须且只能启用一个模型，"
            f"当前启用数量：{len(enabled)}"
        )
    return enabled[0]
