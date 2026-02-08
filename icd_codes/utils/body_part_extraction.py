from __future__ import annotations

import asyncio
from typing import Iterable, Any

import polars as pl
from pydantic import BaseModel, Field

from icd_codes.llm.async_openai_client import AsyncOpenAIClient
from icd_codes.logger import get_logger

logger = get_logger(__name__)


class BodyPartsResponse(BaseModel):
    note_body_parts: list[str] = Field(default_factory=list)
    dialogue_body_parts: list[str] = Field(default_factory=list)
    icd10_desc_body_parts: list[str] = Field(default_factory=list)


_PROMPT_TEMPLATE = """You are extracting anatomical body parts from medical text.
Return a JSON object with exactly these keys:
- note_body_parts
- dialogue_body_parts
- icd10_desc_body_parts

Rules:
- Each value is a list of unique body part strings found in the corresponding field.
- Use free text strings exactly as they appear (normalized to lowercase is fine).
- If no body parts are present, return an empty list for that field.
- Return only JSON. No extra text.

Note:
{note}

Dialogue:
{dialogue}

ICD10_desc:
{icd10_desc}
"""


async def _extract_body_parts_for_row(
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    note: Any,
    dialogue: Any,
    icd10_desc: Any,
) -> BodyPartsResponse:
    prompt = _PROMPT_TEMPLATE.format(
        note=str(note) if note is not None else "",
        dialogue=str(dialogue) if dialogue is not None else "",
        icd10_desc=str(icd10_desc) if icd10_desc is not None else "",
    )

    async with semaphore:
        response = await client.generate(prompt, response_model=BodyPartsResponse)
        content = response.content
        if isinstance(content, BodyPartsResponse):
            return content
        return BodyPartsResponse.model_validate(content)


async def extract_body_parts_dataframe(
    df: pl.DataFrame,
    model_name: str,
    *,
    concurrency: int = 20,
    batch_size: int = 200,
) -> pl.DataFrame:
    """
    Extract body parts from Note, Dialogue, and ICD10_desc per row (single call per row).
    Adds Note_body_parts, Dialogue_body_parts, ICD10_desc_body_parts columns.
    """
    if not {" Note", "Dialogue", "ICD10_desc"}.issubset(set(df.columns)):
        missing = {" Note", "Dialogue", "ICD10_desc"} - set(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    client = AsyncOpenAIClient(model_name=model_name)
    semaphore = asyncio.Semaphore(concurrency)

    note_parts: list[list[str]] = []
    dialogue_parts: list[list[str]] = []
    icd10_desc_parts: list[list[str]] = []

    rows_iter: Iterable[dict[str, Any]] = df.select(
        [" Note", "Dialogue", "ICD10_desc"]
    ).iter_rows(named=True)

    batch: list[asyncio.Task[BodyPartsResponse]] = []

    async def flush_batch() -> None:
        if not batch:
            return
        results = await asyncio.gather(*batch, return_exceptions=True)
        batch.clear()
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Body part extraction failed: {result}")
                note_parts.append([])
                dialogue_parts.append([])
                icd10_desc_parts.append([])
                continue
            note_parts.append(result.note_body_parts)
            dialogue_parts.append(result.dialogue_body_parts)
            icd10_desc_parts.append(result.icd10_desc_body_parts)

    for row in rows_iter:
        task = asyncio.create_task(
            _extract_body_parts_for_row(
                client,
                semaphore,
                row.get(" Note"),
                row.get("Dialogue"),
                row.get("ICD10_desc"),
            )
        )
        batch.append(task)
        if len(batch) >= batch_size:
            await flush_batch()

    await flush_batch()

    return df.with_columns(
        [
            pl.Series(name="Note_body_parts", values=note_parts),
            pl.Series(name="Dialogue_body_parts", values=dialogue_parts),
            pl.Series(name="ICD10_desc_body_parts", values=icd10_desc_parts),
        ]
    )
