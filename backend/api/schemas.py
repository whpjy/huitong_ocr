"""Response models for the web demo API."""

from pydantic import BaseModel, Field


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


class ApplicationSourceFile(BaseModel):
    name: str
    material_code: str


class ApplicationSourceImageGroup(BaseModel):
    material: str
    label: str
    material_code: str
    files: list[ApplicationSourceFile]


class ApplicationSourceImagesResponse(BaseModel):
    application_no: str
    groups: list[ApplicationSourceImageGroup]


class ApplicationDocumentResult(BaseModel):
    document_type: str
    document_name: str
    instance_id: str
    source_files: list[str]
    source_file_refs: list[ApplicationSourceFile] = Field(default_factory=list)
    fields: dict[str, str]
    pipeline_type: str
    provider: str
    model: str
    elapsed_seconds: float
    pairing_confidence: str | None = None
    person_id: str | None = None
    vehicle_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ApplicationDocuments(BaseModel):
    id_cards: list[ApplicationDocumentResult]
    driver_licenses: list[ApplicationDocumentResult]
    vehicle_licenses: list[ApplicationDocumentResult]


class ApplicationValidationResult(BaseModel):
    field: str
    left_instance_id: str
    right_instance_id: str
    status: str
    severity: str
    similarity: float
    message: str


class ApplicationRecognitionError(BaseModel):
    document_type: str
    source_files: list[str]
    error: str


class ApplicationRecognitionSummary(BaseModel):
    id_card_count: int
    driver_license_count: int
    vehicle_license_count: int
    person_count: int
    duplicate_file_count: int
    error_count: int
    missing_documents: list[str]
    elapsed_seconds: float


class ApplicationRecognitionResponse(BaseModel):
    application_no: str
    status: str
    documents: ApplicationDocuments
    validations: list[ApplicationValidationResult]
    errors: list[ApplicationRecognitionError]
    summary: ApplicationRecognitionSummary
