"""Common contract and HTTP-session handling for OCR providers."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from ..config import ProviderConfig
from ..models import ProviderResult


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class OCRProvider(ABC):
    """Provider-specific request/response behavior behind one interface."""

    original_prefix = "原文件"

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._thread_local = threading.local()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def supported_extensions(self) -> set[str]:
        extensions = set(IMAGE_EXTENSIONS)
        if self.config.supports_pdf:
            extensions.add(".pdf")
        return extensions

    def get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            # 局域网 OCR 请求不应被系统 HTTP_PROXY 转发。
            session.trust_env = False
            self._thread_local.session = session
        return session

    @abstractmethod
    def check_health(self, timeout: float | None = None) -> None:
        """Raise when the configured OCR service is not ready."""

    @abstractmethod
    def recognize(self, source_path: Path) -> ProviderResult:
        """Recognize one supported file or raise a concrete exception."""

