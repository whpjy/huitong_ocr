"""Load and validate OpenAI-compatible LLM provider configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


class LLMConfigError(ValueError):
    """Raised when the LLM configuration is missing or inconsistent."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LLMConfigError(f"{location} 必须是对象")
    return value


def _string(value: Any, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise LLMConfigError(f"{location} 必须是字符串")
    result = value.strip()
    if not result and not allow_empty:
        raise LLMConfigError(f"{location} 不能为空")
    return result


def _positive_number(value: Any, location: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise LLMConfigError(f"{location} 必须大于 0")
    return float(value)


def _positive_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LLMConfigError(f"{location} 必须是大于等于 1 的整数")
    return value


def _ratio(value: Any, location: str, *, allow_zero: bool = False) -> float:
    lower_bound = 0 if allow_zero else 0.0
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < lower_bound
        or (not allow_zero and value == 0)
        or value > 1
    ):
        qualifier = "0 到 1" if allow_zero else "大于 0 且不超过 1"
        raise LLMConfigError(f"{location} 必须{qualifier}")
    return float(value)


def _endpoint(value: Any, location: str) -> str:
    endpoint = _string(value, location)
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


@dataclass(frozen=True)
class DocumentCropConfig:
    """Safe local document-boundary detection before vision inference."""

    enabled: bool = False
    perspective_correction: bool = True
    detection_max_side: int = 1600
    min_area_ratio: float = 0.10
    max_area_ratio: float = 0.70
    min_rectangularity: float = 0.65
    max_outside_edge_ratio: float = 0.15
    min_aspect_ratio: float = 1.15
    max_aspect_ratio: float = 3.8
    padding_ratio: float = 0.04
    jpeg_quality: int = 95

    @classmethod
    def from_mapping(
        cls,
        raw_value: Any,
        location: str,
    ) -> "DocumentCropConfig":
        raw = _mapping(raw_value, location)
        enabled = raw.get("enabled", False)
        perspective = raw.get("perspective_correction", True)
        if not isinstance(enabled, bool):
            raise LLMConfigError(f"{location}.enabled 必须是布尔值")
        if not isinstance(perspective, bool):
            raise LLMConfigError(
                f"{location}.perspective_correction 必须是布尔值"
            )
        detection_max_side = _positive_int(
            raw.get("detection_max_side", 1600),
            f"{location}.detection_max_side",
        )
        jpeg_quality = _positive_int(
            raw.get("jpeg_quality", 95),
            f"{location}.jpeg_quality",
        )
        if detection_max_side < 320:
            raise LLMConfigError(
                f"{location}.detection_max_side 不能小于 320"
            )
        if jpeg_quality > 100:
            raise LLMConfigError(f"{location}.jpeg_quality 不能超过 100")
        min_area_ratio = _ratio(
            raw.get("min_area_ratio", 0.10),
            f"{location}.min_area_ratio",
        )
        max_area_ratio = _ratio(
            raw.get("max_area_ratio", 0.70),
            f"{location}.max_area_ratio",
        )
        if min_area_ratio >= max_area_ratio:
            raise LLMConfigError(
                f"{location}.min_area_ratio 必须小于 max_area_ratio"
            )
        min_aspect = _positive_number(
            raw.get("min_aspect_ratio", 1.15),
            f"{location}.min_aspect_ratio",
        )
        max_aspect = _positive_number(
            raw.get("max_aspect_ratio", 3.8),
            f"{location}.max_aspect_ratio",
        )
        if min_aspect >= max_aspect:
            raise LLMConfigError(
                f"{location}.min_aspect_ratio 必须小于 max_aspect_ratio"
            )
        return cls(
            enabled=enabled,
            perspective_correction=perspective,
            detection_max_side=detection_max_side,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_rectangularity=_ratio(
                raw.get("min_rectangularity", 0.65),
                f"{location}.min_rectangularity",
            ),
            max_outside_edge_ratio=_ratio(
                raw.get("max_outside_edge_ratio", 0.15),
                f"{location}.max_outside_edge_ratio",
                allow_zero=True,
            ),
            min_aspect_ratio=min_aspect,
            max_aspect_ratio=max_aspect,
            padding_ratio=_ratio(
                raw.get("padding_ratio", 0.04),
                f"{location}.padding_ratio",
                allow_zero=True,
            ),
            jpeg_quality=jpeg_quality,
        )


@dataclass(frozen=True)
class LLMProviderConfig:
    """One concrete model endpoint and its runtime options."""

    name: str
    type: str
    enabled: bool
    base_url: str
    endpoint: str
    api_key: str
    api_key_env: str
    model: str
    timeout: float
    max_attempts: int
    concurrency: int
    temperature: float
    response_format: str | None
    extra_body: dict[str, Any]
    input_mode: str = "text"
    display_name: str = ""
    vision_content_order: str = "images_first"
    vision_max_image_side: int | None = None
    vision_response_adapter: str = "json"

    @property
    def service_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.endpoint}"

    def resolve_api_key(self) -> str:
        """Prefer the environment and use the YAML value as local fallback."""

        environment_value = (
            os.getenv(self.api_key_env, "").strip()
            if self.api_key_env
            else ""
        )
        return environment_value or self.api_key

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw_value: Any,
        location: str,
    ) -> "LLMProviderConfig":
        raw = _mapping(raw_value, location)
        provider_type = _string(raw.get("type"), f"{location}.type")
        if provider_type != "openai_compatible":
            raise LLMConfigError(
                f"{location}.type 不支持：{provider_type}；"
                "当前仅支持 openai_compatible"
            )

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise LLMConfigError(f"{location}.enabled 必须是布尔值")

        input_mode = _string(
            raw.get("input_mode", "text"),
            f"{location}.input_mode",
        )
        if input_mode not in {"text", "vision"}:
            raise LLMConfigError(
                f"{location}.input_mode 仅支持 text 或 vision"
            )

        vision_content_order = _string(
            raw.get("vision_content_order", "images_first"),
            f"{location}.vision_content_order",
        )
        if vision_content_order not in {"images_first", "text_first"}:
            raise LLMConfigError(
                f"{location}.vision_content_order 仅支持 "
                "images_first 或 text_first"
            )

        max_image_side_value = raw.get("vision_max_image_side")
        vision_max_image_side = (
            None
            if max_image_side_value in (None, "")
            else _positive_int(
                max_image_side_value,
                f"{location}.vision_max_image_side",
            )
        )
        if vision_max_image_side is not None and vision_max_image_side < 320:
            raise LLMConfigError(
                f"{location}.vision_max_image_side 不能小于 320"
            )
        vision_response_adapter = _string(
            raw.get("vision_response_adapter", "json"),
            f"{location}.vision_response_adapter",
        )
        if vision_response_adapter not in {
            "json",
            "labeled_text",
            "hunyuan_ocr",
        }:
            raise LLMConfigError(
                f"{location}.vision_response_adapter 仅支持 "
                "json、labeled_text 或 hunyuan_ocr"
            )
        if (
            vision_response_adapter in {"labeled_text", "hunyuan_ocr"}
            and input_mode != "vision"
        ):
            raise LLMConfigError(
                f"{location}.vision_response_adapter={vision_response_adapter} "
                "仅适用于 vision"
            )
        temperature = raw.get("temperature", 0.0)
        if not isinstance(temperature, (int, float)) or isinstance(
            temperature, bool
        ):
            raise LLMConfigError(f"{location}.temperature 必须是数字")

        response_format_value = raw.get("response_format", "json_object")
        if response_format_value in (None, False, ""):
            response_format = None
        elif response_format_value == "json_object":
            response_format = "json_object"
        else:
            raise LLMConfigError(
                f"{location}.response_format 仅支持 json_object 或空值"
            )

        extra_body_raw = _mapping(
            raw.get("extra_body", {}),
            f"{location}.extra_body",
        )
        if not all(isinstance(key, str) for key in extra_body_raw):
            raise LLMConfigError(f"{location}.extra_body 的 key 必须是字符串")
        reserved_keys = {
            "model",
            "messages",
            "temperature",
            "response_format",
        }
        conflicts = reserved_keys.intersection(extra_body_raw)
        if conflicts:
            names = "、".join(sorted(conflicts))
            raise LLMConfigError(
                f"{location}.extra_body 不能覆盖标准参数：{names}"
            )
        extra_body = dict(extra_body_raw)
        try:
            json.dumps(extra_body)
        except (TypeError, ValueError) as exc:
            raise LLMConfigError(
                f"{location}.extra_body 必须可以序列化为 JSON"
            ) from exc

        return cls(
            name=name,
            type=provider_type,
            enabled=enabled,
            input_mode=input_mode,
            base_url=_string(raw.get("base_url"), f"{location}.base_url"),
            endpoint=_endpoint(raw.get("endpoint"), f"{location}.endpoint"),
            api_key=_string(
                raw.get("api_key", ""),
                f"{location}.api_key",
                allow_empty=True,
            ),
            api_key_env=_string(
                raw.get("api_key_env", ""),
                f"{location}.api_key_env",
                allow_empty=True,
            ),
            model=_string(raw.get("model"), f"{location}.model"),
            timeout=_positive_number(raw.get("timeout"), f"{location}.timeout"),
            max_attempts=_positive_int(
                raw.get("max_attempts"),
                f"{location}.max_attempts",
            ),
            concurrency=_positive_int(
                raw.get("concurrency"),
                f"{location}.concurrency",
            ),
            temperature=float(temperature),
            response_format=response_format,
            extra_body=extra_body,
            display_name=_string(
                raw.get("display_name", ""),
                f"{location}.display_name",
                allow_empty=True,
            ),
            vision_content_order=vision_content_order,
            vision_max_image_side=vision_max_image_side,
            vision_response_adapter=vision_response_adapter,
        )


@dataclass(frozen=True)
class LLMManagerConfig:
    """Validated top-level LLM configuration."""

    config_path: Path
    backend_root: Path
    output_root: Path
    active_provider: str
    providers: dict[str, LLMProviderConfig]
    prompts: dict[str, str]
    document_crop: DocumentCropConfig

    def get_provider(self, name: str | None = None) -> LLMProviderConfig:
        provider_name = name or self.active_provider
        provider = self.providers.get(provider_name)
        if provider is None:
            raise LLMConfigError(f"未配置 LLM Provider：{provider_name}")
        if not provider.enabled:
            raise LLMConfigError(f"LLM Provider 已禁用：{provider_name}")
        return provider


def load_llm_config(path: str | Path | None = None) -> LLMManagerConfig:
    """Read the LLM YAML file and resolve backend-relative paths."""

    default_path = Path(__file__).resolve().parents[1] / "config" / "llm.yaml"
    config_path = Path(path).expanduser().resolve() if path else default_path
    if not config_path.is_file():
        raise LLMConfigError(f"LLM 配置文件不存在：{config_path}")

    try:
        raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LLMConfigError(f"LLM YAML 解析失败：{exc}") from exc

    raw = _mapping(raw_value, "root")
    if raw.get("version") != 1:
        raise LLMConfigError("仅支持 LLM 配置版本 version: 1")

    backend_root = config_path.parent.parent.resolve()
    paths = _mapping(raw.get("paths"), "paths")
    output_value = Path(
        _string(paths.get("output_root"), "paths.output_root")
    )
    output_root = (
        output_value if output_value.is_absolute() else backend_root / output_value
    ).resolve()

    section_name = "multimodal" if "multimodal" in raw else "llm"
    llm = _mapping(raw.get(section_name), section_name)
    active_provider = _string(
        llm.get("active_provider"),
        f"{section_name}.active_provider",
    )
    providers_raw = _mapping(
        llm.get("providers"),
        f"{section_name}.providers",
    )
    providers = {
        str(name): LLMProviderConfig.from_mapping(
            str(name),
            value,
            f"{section_name}.providers.{name}",
        )
        for name, value in providers_raw.items()
    }
    if active_provider not in providers:
        raise LLMConfigError(
            f"{section_name}.active_provider 未在 providers 中配置："
            f"{active_provider}"
        )
    if not providers[active_provider].enabled:
        raise LLMConfigError(
            f"{section_name}.active_provider 已被禁用：{active_provider}"
        )

    prompts_raw = _mapping(
        llm.get("prompts", {}),
        f"{section_name}.prompts",
    )
    prompts = {
        str(name): _string(value, f"{section_name}.prompts.{name}")
        for name, value in prompts_raw.items()
    }
    preprocessing = _mapping(
        llm.get("preprocessing", {}),
        f"{section_name}.preprocessing",
    )
    document_crop = DocumentCropConfig.from_mapping(
        preprocessing.get("document_crop", {}),
        f"{section_name}.preprocessing.document_crop",
    )

    return LLMManagerConfig(
        config_path=config_path,
        backend_root=backend_root,
        output_root=output_root,
        active_provider=active_provider,
        providers=providers,
        prompts=prompts,
        document_crop=document_crop,
    )


def load_multimodal_config(
    path: str | Path | None = None,
) -> LLMManagerConfig:
    """Load the dedicated multimodal model configuration file."""

    default_path = Path(__file__).resolve().parents[1] / "config" / "multimodal.yaml"
    config = load_llm_config(path or default_path)
    hunyuan_base_url = os.getenv("HUNYUAN_OCR_BASE_URL", "").strip()
    if hunyuan_base_url and "hunyuan_ocr" in config.providers:
        providers = dict(config.providers)
        providers["hunyuan_ocr"] = replace(
            providers["hunyuan_ocr"],
            base_url=hunyuan_base_url,
        )
        config = replace(config, providers=providers)
    required_prompts = {
        "general_system",
        "general_user",
        "hunyuan_general_system",
        "hunyuan_general_user",
        "document_system_prefix",
        "document_user_prefix",
    }
    missing = sorted(required_prompts.difference(config.prompts))
    if missing:
        raise LLMConfigError(
            "multimodal.prompts 缺少配置：" + "、".join(missing)
        )
    return config
