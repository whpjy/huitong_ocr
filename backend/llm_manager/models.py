"""Result models for one extraction and an OCR-JSON batch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExtractionResult:
    application_no: str
    success: bool
    status: str
    fields: dict[str, str]
    elapsed: float
    error: str = ""
    records: list[dict[str, str]] | None = None
    quality_check: str = ""
    quality_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        records = self.records or [self.fields]
        payload = {
            "application_no": self.application_no,
            "success": self.success,
            "status": self.status,
            "fields": self.fields,
            "records": records,
            "elapsed_seconds": round(self.elapsed, 6),
            "error": self.error or None,
        }
        if self.quality_check:
            payload["quality_check"] = self.quality_check
            payload["quality_detail"] = self.quality_detail
        return payload


@dataclass
class BatchExtractionResult:
    document_key: str
    document_name: str
    provider: str
    model: str
    source_file: Path
    output_file: Path
    elapsed: float
    results: list[ExtractionResult]
    skipped: int = 0
    error_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        succeeded = sum(item.success for item in self.results)
        failed = len(self.results) - succeeded
        empty = sum(item.status == "empty_ocr" for item in self.results)
        return {
            "document": {
                "key": self.document_key,
                "name": self.document_name,
            },
            "provider": self.provider,
            "model": self.model,
            "source_file": str(self.source_file),
            "output_file": str(self.output_file),
            "summary": {
                "total": len(self.results),
                "succeeded": succeeded,
                "failed": failed,
                "empty_ocr": empty,
                "skipped": self.skipped,
                "elapsed_seconds": round(self.elapsed, 6),
            },
            "applications": [item.to_dict() for item in self.results],
            "error_file": str(self.error_file) if self.error_file else None,
        }
