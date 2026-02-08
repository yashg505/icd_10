from typing import List, Literal

from pydantic import BaseModel, Field


class ClinicalEntities(BaseModel):
    primary_diagnosis: str = Field(
        description="The main clinical assessment or suspected condition."
    )
    laterality: Literal["Left", "Right", "Bilateral", "Unspecified"] = Field(
        description="Side of the body affected."
    )
    anatomy: str = Field(
        description="Specific body part or organ system involved."
    )
    symptoms_findings: List[str] = Field(
        description="Key symptoms or clinical findings."
    )
    objective_evidence: List[str] = Field(
        description="Exam results or objective evidence."
    )
