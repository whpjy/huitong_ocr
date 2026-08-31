"""Configuration for application-number document recognition."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ApplicationConfigError(ValueError):
    """Raised when the application recognition configuration is invalid."""


@dataclass(frozen=True)
class ApplicationRecognitionConfig:
    data_root: Path
    material_codes: dict[str, str]
    max_workers: int = 3


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApplicationConfigError(f"{location} 必须是对象")
    return value


def load_application_config(
    path: str | Path | None = None,
) -> ApplicationRecognitionConfig:
    config_path = Path(path) if path else (
        Path(__file__).resolve().parents[1] / "config" / "applications.yaml"
    )
    if not config_path.is_file():
        raise ApplicationConfigError(f"申请单配置文件不存在：{config_path}")
    try:
        raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ApplicationConfigError(f"申请单 YAML 解析失败：{exc}") from exc

    raw = _mapping(raw_value, "root")
    if raw.get("version") != 1:
        raise ApplicationConfigError("仅支持申请单配置版本 version: 1")
    paths = _mapping(raw.get("paths"), "paths")
    applications = _mapping(raw.get("applications"), "applications")
    materials = _mapping(applications.get("materials"), "applications.materials")

    configured_root = os.getenv("APPLICATION_DATA_ROOT", "").strip()
    raw_root = configured_root or str(paths.get("data_root") or "").strip()
    if not raw_root:
        raise ApplicationConfigError("paths.data_root 不能为空")
    data_root = Path(raw_root).expanduser()
    if not data_root.is_absolute():
        data_root = config_path.resolve().parent / data_root
    data_root = data_root.resolve()

    required = {
        "id_card_front",
        "id_card_back",
        "driver_license",
        "vehicle_license",
    }
    codes = {
        str(name): str(code).strip()
        for name, code in materials.items()
        if str(code).strip()
    }
    missing = sorted(required - set(codes))
    if missing:
        raise ApplicationConfigError(
            f"applications.materials 缺少配置：{', '.join(missing)}"
        )

    try:
        max_workers = int(applications.get("max_workers", 3))
    except (TypeError, ValueError) as exc:
        raise ApplicationConfigError("applications.max_workers 必须是整数") from exc
    if not 1 <= max_workers <= 8:
        raise ApplicationConfigError("applications.max_workers 必须在 1 到 8 之间")

    return ApplicationRecognitionConfig(
        data_root=data_root,
        material_codes=codes,
        max_workers=max_workers,
    )
