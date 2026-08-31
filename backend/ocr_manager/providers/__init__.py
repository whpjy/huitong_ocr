"""OCR provider factory."""

from __future__ import annotations

from ..config import ConfigError, ProviderConfig
from .base import OCRProvider
from .pp_ocrv6 import PPOCRv6Provider


def create_provider(config: ProviderConfig) -> OCRProvider:
    if config.type == "pp_ocrv6":
        return PPOCRv6Provider(config)
    raise ConfigError(f"不支持的 OCR Provider 类型：{config.type}")


__all__ = [
    "OCRProvider",
    "PPOCRv6Provider",
    "create_provider",
]
