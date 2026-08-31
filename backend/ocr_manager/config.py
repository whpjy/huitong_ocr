"""Load and validate OCR providers and document profiles from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when the OCR configuration is missing or inconsistent."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} 必须是对象")
    return value


def _non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} 必须是非空字符串")
    return value.strip()


def _positive_number(value: Any, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{location} 必须大于 0")
    return float(value)


def _non_negative_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{location} 必须是大于等于 0 的整数")
    return value


def _positive_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{location} 必须是大于等于 1 的整数")
    return value


def _endpoint(value: Any, location: str) -> str:
    endpoint = _non_empty_string(value, location)
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


@dataclass(frozen=True)
class ProviderConfig:
    """One concrete OCR service profile."""

    name: str
    type: str
    enabled: bool
    base_url: str
    endpoint: str
    health_endpoint: str
    timeout: float
    retries: int
    concurrency: int
    supports_pdf: bool
    max_image_side: int | None = None

    @property
    def service_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.endpoint}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.health_endpoint}"

    @classmethod
    def from_mapping(
        cls, name: str, raw_value: Any, location: str
    ) -> "ProviderConfig":
        raw = _mapping(raw_value, location)
        provider_type = _non_empty_string(raw.get("type"), f"{location}.type")
        if provider_type != "pp_ocrv6":
            raise ConfigError(
                f"{location}.type 不支持：{provider_type}；"
                "精简服务仅支持 pp_ocrv6"
            )
        enabled = raw.get("enabled", True)
        supports_pdf = raw.get("supports_pdf", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{location}.enabled 必须是布尔值")
        if not isinstance(supports_pdf, bool):
            raise ConfigError(f"{location}.supports_pdf 必须是布尔值")
        max_image_side_value = raw.get("max_image_side")
        max_image_side = (
            None
            if max_image_side_value in (None, "")
            else _positive_int(
                max_image_side_value,
                f"{location}.max_image_side",
            )
        )
        if max_image_side is not None and max_image_side < 320:
            raise ConfigError(f"{location}.max_image_side 不能小于 320")
        return cls(
            name=name,
            type=provider_type,
            enabled=enabled,
            base_url=_non_empty_string(raw.get("base_url"), f"{location}.base_url"),
            endpoint=_endpoint(raw.get("endpoint"), f"{location}.endpoint"),
            health_endpoint=_endpoint(
                raw.get("health_endpoint", "/health/ready"),
                f"{location}.health_endpoint",
            ),
            timeout=_positive_number(raw.get("timeout"), f"{location}.timeout"),
            retries=_non_negative_int(raw.get("retries"), f"{location}.retries"),
            concurrency=_positive_int(
                raw.get("concurrency"), f"{location}.concurrency"
            ),
            supports_pdf=supports_pdf,
            max_image_side=max_image_side,
        )


@dataclass(frozen=True)
class DocumentConfig:
    """Directory and material-code rules for one document type."""

    key: str
    name: str
    target_codes: tuple[str, ...]
    default_input_root: Path
    supported_providers: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        key: str,
        raw_value: Any,
        location: str,
        workspace_root: Path,
    ) -> "DocumentConfig":
        raw = _mapping(raw_value, location)
        codes_value = raw.get("target_codes")
        providers_value = raw.get("supported_providers")
        if not isinstance(codes_value, list) or not codes_value:
            raise ConfigError(f"{location}.target_codes 必须是非空数组")
        if not isinstance(providers_value, list) or not providers_value:
            raise ConfigError(f"{location}.supported_providers 必须是非空数组")
        codes = tuple(
            _non_empty_string(item, f"{location}.target_codes[]")
            for item in codes_value
        )
        providers = tuple(
            _non_empty_string(item, f"{location}.supported_providers[]")
            for item in providers_value
        )
        input_value = Path(
            _non_empty_string(
                raw.get("default_input_root"), f"{location}.default_input_root"
            )
        )
        input_root = (
            input_value if input_value.is_absolute() else workspace_root / input_value
        ).resolve()
        return cls(
            key=key,
            name=_non_empty_string(raw.get("name"), f"{location}.name"),
            target_codes=codes,
            default_input_root=input_root,
            supported_providers=providers,
        )


@dataclass(frozen=True)
class OrientationConfig:
    """Local document orientation classifier settings."""

    enabled: bool
    model: str
    device: str


@dataclass(frozen=True)
class OCRManagerConfig:
    """Validated top-level configuration."""

    config_path: Path
    backend_root: Path
    workspace_root: Path
    output_root: Path
    active_provider: str
    providers: dict[str, ProviderConfig]
    documents: dict[str, DocumentConfig]
    orientation: OrientationConfig

    def get_provider(self, name: str | None = None) -> ProviderConfig:
        provider_name = name or self.active_provider
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ConfigError(f"未配置 OCR Provider：{provider_name}")
        if not provider.enabled:
            raise ConfigError(f"OCR Provider 已禁用：{provider_name}")
        return provider

    def get_document(self, key: str) -> DocumentConfig:
        document = self.documents.get(key)
        if document is None:
            available = "、".join(sorted(self.documents))
            raise ConfigError(f"未配置证件类型：{key}；可选值：{available}")
        return document

    def validate_pair(
        self, document: DocumentConfig, provider: ProviderConfig
    ) -> None:
        if provider.name not in document.supported_providers:
            raise ConfigError(
                f"{document.name} 不支持 Provider {provider.name}；"
                f"支持：{'、'.join(document.supported_providers)}"
            )


def load_config(path: str | Path | None = None) -> OCRManagerConfig:
    """Read one YAML file and resolve all configured paths."""

    default_path = Path(__file__).resolve().parents[1] / "config" / "ocr.yaml"
    config_path = Path(path).expanduser().resolve() if path else default_path
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在：{config_path}")

    try:
        raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败：{exc}") from exc
    raw = _mapping(raw_value, "root")
    if raw.get("version") != 1:
        raise ConfigError("仅支持配置版本 version: 1")

    backend_root = config_path.parent.parent.resolve()
    paths_raw = _mapping(raw.get("paths"), "paths")
    workspace_value = Path(
        _non_empty_string(paths_raw.get("workspace_root"), "paths.workspace_root")
    )
    workspace_root = (
        workspace_value
        if workspace_value.is_absolute()
        else backend_root / workspace_value
    ).resolve()
    output_value = Path(
        _non_empty_string(paths_raw.get("output_root"), "paths.output_root")
    )
    output_root = (
        output_value if output_value.is_absolute() else backend_root / output_value
    ).resolve()

    ocr_raw = _mapping(raw.get("ocr"), "ocr")
    active_provider = _non_empty_string(
        ocr_raw.get("active_provider"), "ocr.active_provider"
    )
    providers_raw = _mapping(ocr_raw.get("providers"), "ocr.providers")
    providers = {
        str(name): ProviderConfig.from_mapping(
            str(name), value, f"ocr.providers.{name}"
        )
        for name, value in providers_raw.items()
    }
    ppocr_base_url = os.getenv("PPOCR_BASE_URL", "").strip()
    if ppocr_base_url and "pp_ocrv6" in providers:
        providers["pp_ocrv6"] = replace(
            providers["pp_ocrv6"],
            base_url=ppocr_base_url,
        )
    if active_provider not in providers:
        raise ConfigError(
            f"ocr.active_provider 未在 providers 中配置：{active_provider}"
        )
    if not providers[active_provider].enabled:
        raise ConfigError(f"ocr.active_provider 已被禁用：{active_provider}")

    orientation_raw = _mapping(
        ocr_raw.get("orientation", {}),
        "ocr.orientation",
    )
    orientation_enabled = orientation_raw.get("enabled", True)
    if not isinstance(orientation_enabled, bool):
        raise ConfigError("ocr.orientation.enabled must be a boolean")
    orientation = OrientationConfig(
        enabled=orientation_enabled,
        model=_non_empty_string(
            orientation_raw.get("model", "PP-LCNet_x1_0_doc_ori"),
            "ocr.orientation.model",
        ),
        device=_non_empty_string(
            orientation_raw.get("device", "cpu"),
            "ocr.orientation.device",
        ),
    )

    documents_raw = _mapping(raw.get("documents"), "documents")
    documents = {
        str(key): DocumentConfig.from_mapping(
            str(key), value, f"documents.{key}", workspace_root
        )
        for key, value in documents_raw.items()
    }
    if not documents:
        raise ConfigError("documents 至少需要配置一个证件类型")

    for document in documents.values():
        unknown = set(document.supported_providers) - set(providers)
        if unknown:
            raise ConfigError(
                f"documents.{document.key}.supported_providers "
                f"包含未配置 Provider：{'、'.join(sorted(unknown))}"
            )

    return OCRManagerConfig(
        config_path=config_path,
        backend_root=backend_root,
        workspace_root=workspace_root,
        output_root=output_root,
        active_provider=active_provider,
        providers=providers,
        documents=documents,
        orientation=orientation,
    )
