import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm

from icd_codes.models.clinical import ClinicalEntities

load_dotenv()

_DEFAULT_MODEL = "gpt-5-mini"
_DEFAULT_CONCURRENCY = 20

_SYSTEM_PROMPT = """
Extract clinical entities from the medical note.
Do not infer anything not present in the text. Be Brief and precise and avoid unnecessary noise.
Focus on the most critical information relevant to diagnosis.
"""

_client = AsyncOpenAI()


def _format_entities(parsed: ClinicalEntities) -> str:
    return (
        f"{parsed.primary_diagnosis}, {parsed.laterality} {parsed.anatomy}. "
        f"{', '.join(parsed.symptoms_findings)}. "
        f"{', '.join(parsed.objective_evidence)}."
    )


async def summarize_notes(
    notes: list[str],
    model_name: str = _DEFAULT_MODEL,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[str]:
    """
    Reusable summarization utility for a list of clinical notes.
    Returns one summary string per input note.
    """
    semaphore = asyncio.Semaphore(concurrency)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def _call(note: str) -> str:
        async with semaphore:
            response = await _client.responses.parse(
                model=model_name,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": str(note)},
                ],
                text_format=ClinicalEntities,
            )
            parsed: ClinicalEntities = response.output_parsed
            return _format_entities(parsed)

    tasks = [_call(str(n) if n is not None else "") for n in notes]
    return await tqdm.gather(*tasks)


@dataclass
class NoteSummarizer:
    model_name: str
    concurrency: int = _DEFAULT_CONCURRENCY

    async def summarize(self, notes: list[str]) -> list[str]:
        return await summarize_notes(
            notes=notes,
            model_name=self.model_name,
            concurrency=self.concurrency,
        )

    def summarize_sync(self, notes: list[str]) -> list[str]:
        return asyncio.run(self.summarize(notes))
