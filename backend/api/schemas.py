"""Response models for the web demo API."""

from pydantic import BaseModel


class ExtractionTiming(BaseModel):
    ocr_seconds: float
    llm_seconds: float
    total_seconds: float


class DocumentRecognitionResponse(BaseModel):
    filename: str
    document_type: str
    document_name: str
    pipeline_type: str
    provider: str
    model: str
    source_text: str
    fields: dict[str, str]
    timing: ExtractionTiming


class MobileRecognitionConfigResponse(BaseModel):
    name: str
    model_key: str
    label: str
    pipeline_type: str
    image_quality_enabled: bool
