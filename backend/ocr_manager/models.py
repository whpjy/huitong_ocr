"""Data models shared by scanning, providers, and batch execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRTask:
    index: int
    application_no: str
    document_code: str
    source_path: Path
    output_dir: Path
    output_stem: str
    duplicate_of: int | None = None


@dataclass(frozen=True)
class ProviderResult:
    text: str
    page_texts: tuple[str, ...]
    visualizations: tuple[bytes, ...]
    tokens: tuple[dict[str, Any], ...] = ()

    @property
    def page_count(self) -> int:
        return len(self.page_texts)


@dataclass
class TaskResult:
    task: OCRTask
    success: bool
    text: str
    elapsed: float
    normalized_text: str = ""
    page_count: int = 0
    error: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.task.index,
            "application_no": self.task.application_no,
            "document_code": self.task.document_code,
            "source_path": str(self.task.source_path),
            "success": self.success,
            "text": self.text,
            "normalized_text": self.normalized_text or self.text,
            "elapsed_seconds": round(self.elapsed, 6),
            "page_count": self.page_count,
            "error": self.error or None,
            "artifacts": self.artifacts,
        }


@dataclass(frozen=True)
class ScanResult:
    input_root: Path
    run_dir: Path
    applications: tuple[Path, ...]
    tasks: tuple[OCRTask, ...]
    tasks_by_application: dict[str, tuple[OCRTask, ...]]

    def summary(self) -> dict[str, Any]:
        counts_by_code: dict[str, int] = {}
        image_count = 0
        pdf_count = 0
        for task in self.tasks:
            counts_by_code[task.document_code] = (
                counts_by_code.get(task.document_code, 0) + 1
            )
            if task.source_path.suffix.lower() == ".pdf":
                pdf_count += 1
            else:
                image_count += 1
        return {
            "input_root": str(self.input_root),
            "run_dir": str(self.run_dir),
            "application_count": len(self.applications),
            "file_count": len(self.tasks),
            "image_count": image_count,
            "pdf_count": pdf_count,
            "counts_by_code": counts_by_code,
        }


@dataclass
class RunResult:
    document_key: str
    document_name: str
    provider: str
    input_root: Path
    output_dir: Path
    elapsed: float
    task_results: list[TaskResult]
    applications: list[dict[str, Any]]
    result_file: Path | None = None
    error_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        succeeded = sum(item.success for item in self.task_results)
        payload = {
            "document": {
                "key": self.document_key,
                "name": self.document_name,
            },
            "provider": self.provider,
            "input_root": str(self.input_root),
            "output_dir": str(self.output_dir),
            "summary": {
                "total": len(self.task_results),
                "succeeded": succeeded,
                "failed": len(self.task_results) - succeeded,
                "elapsed_seconds": round(self.elapsed, 6),
            },
            "applications": self.applications,
            "files": [item.to_dict() for item in self.task_results],
            "result_file": str(self.result_file) if self.result_file else None,
            "error_file": str(self.error_file) if self.error_file else None,
        }
        return payload
