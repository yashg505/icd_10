from pydantic import BaseModel, Field


class ICDPrediction(BaseModel):
    icd_code: str = Field(..., description="ICD-10 code from the provided candidate list.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str


class ICDSelection(BaseModel):
    primary_icd_code: str = Field(..., description="Primary ICD-10 code from the candidate list.")
    predictions: list[ICDPrediction] = Field(default_factory=list)
