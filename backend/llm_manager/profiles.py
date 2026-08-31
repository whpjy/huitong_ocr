"""Load document fields and prompts from extraction_profiles.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .normalizers import has_normalizer


class ProfileConfigError(ValueError):
    """Raised when an extraction profile is invalid."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileConfigError(f"{location} 必须是对象")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileConfigError(f"{location} 必须是非空字符串")
    return value.strip()


@dataclass(frozen=True)
class FieldConfig:
    name: str
    normalizer: str


@dataclass(frozen=True)
class ExtractionProfile:
    key: str
    name: str
    system_prompt: str
    user_prompt_template: str
    empty_values: tuple[str, ...]
    fields: tuple[FieldConfig, ...]
    multi_record: bool = False

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def build_messages(self, ocr_text: str) -> list[dict[str, str]]:
        fields_text = "\n".join(f"- {name}" for name in self.field_names)
        user_prompt = self.user_prompt_template.replace(
            "{{ fields }}",
            fields_text,
        ).replace(
            "{{ ocr_text }}",
            ocr_text,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def normalize(self, data: Mapping[str, Any]) -> dict[str, str]:
        from .normalizers import normalize_value

        return {
            field.name: normalize_value(
                field.normalizer,
                data.get(field.name),
                empty_values=self.empty_values,
            )
            for field in self.fields
        }

    def normalize_records(self, data: Any) -> list[dict[str, str]]:
        """Normalize either the legacy object or a multi-record response."""

        raw_records: Any
        if isinstance(data, Mapping) and isinstance(data.get("records"), list):
            raw_records = data["records"]
        elif isinstance(data, list):
            raw_records = data
        elif isinstance(data, Mapping):
            raw_records = [data]
        else:
            raise ValueError("模型输出必须是 JSON 对象或记录数组")

        records = [
            self.normalize(item)
            for item in raw_records
            if isinstance(item, Mapping)
        ]
        if not records:
            raise ValueError("模型输出没有有效的证件记录")
        return records

    @classmethod
    def from_mapping(
        cls,
        key: str,
        raw_value: Any,
        location: str,
    ) -> "ExtractionProfile":
        raw = _mapping(raw_value, location)
        fields_value = raw.get("fields")
        if not isinstance(fields_value, list) or not fields_value:
            raise ProfileConfigError(f"{location}.fields 必须是非空数组")

        fields: list[FieldConfig] = []
        seen: set[str] = set()
        for index, item in enumerate(fields_value):
            field_raw = _mapping(item, f"{location}.fields[{index}]")
            field_name = _string(
                field_raw.get("name"),
                f"{location}.fields[{index}].name",
            )
            normalizer = _string(
                field_raw.get("normalizer", "text"),
                f"{location}.fields[{index}].normalizer",
            )
            if field_name in seen:
                raise ProfileConfigError(
                    f"{location}.fields 存在重复字段：{field_name}"
                )
            if not has_normalizer(normalizer):
                raise ProfileConfigError(
                    f"{location}.fields[{index}].normalizer "
                    f"不存在：{normalizer}"
                )
            seen.add(field_name)
            fields.append(FieldConfig(field_name, normalizer))

        empty_values_raw = raw.get(
            "empty_values",
            ["", "null", "none", "unknown", "未知", "未识别", "无", "-", "--"],
        )
        if not isinstance(empty_values_raw, list):
            raise ProfileConfigError(f"{location}.empty_values 必须是数组")
        empty_values = tuple(str(value) for value in empty_values_raw)

        user_template = _string(
            raw.get("user_prompt_template"),
            f"{location}.user_prompt_template",
        )
        for placeholder in ("{{ fields }}", "{{ ocr_text }}"):
            if placeholder not in user_template:
                raise ProfileConfigError(
                    f"{location}.user_prompt_template 缺少占位符：{placeholder}"
                )

        return cls(
            key=key,
            name=_string(raw.get("name"), f"{location}.name"),
            system_prompt=_string(
                raw.get("system_prompt"),
                f"{location}.system_prompt",
            ),
            user_prompt_template=user_template,
            empty_values=empty_values,
            fields=tuple(fields),
            multi_record=bool(raw.get("multi_record", False)),
        )


@dataclass(frozen=True)
class ExtractionProfiles:
    config_path: Path
    profiles: dict[str, ExtractionProfile]

    def get(self, key: str) -> ExtractionProfile:
        profile = self.profiles.get(key)
        if profile is None:
            available = "、".join(sorted(self.profiles))
            raise ProfileConfigError(
                f"未配置抽取材料类型：{key}；可选值：{available}"
            )
        return profile


def load_profiles(path: str | Path | None = None) -> ExtractionProfiles:
    default_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "extraction_profiles.yaml"
    )
    config_path = Path(path).expanduser().resolve() if path else default_path
    if not config_path.is_file():
        raise ProfileConfigError(f"字段配置文件不存在：{config_path}")

    try:
        raw_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileConfigError(f"字段 YAML 解析失败：{exc}") from exc
    raw = _mapping(raw_value, "root")
    if raw.get("version") != 1:
        raise ProfileConfigError("仅支持字段配置版本 version: 1")
    profiles_raw = _mapping(raw.get("profiles"), "profiles")
    profiles = {
        str(key): ExtractionProfile.from_mapping(
            str(key),
            value,
            f"profiles.{key}",
        )
        for key, value in profiles_raw.items()
    }
    if not profiles:
        raise ProfileConfigError("profiles 至少需要配置一种材料")
    return ExtractionProfiles(config_path=config_path, profiles=profiles)
