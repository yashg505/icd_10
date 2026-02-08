from __future__ import annotations

import asyncio
import sys
from typing import Any, Iterable

import polars as pl

from icd_codes.llm.async_openai_client import AsyncOpenAIClient
from icd_codes.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = '''
Summarize the provided conversation between patient and doctor in exactly three concise bullet points, capturing the most important topics, medical issues discussed, and any actions or diagnostic plans mentioned.

- Carefully read the entire conversation and identify the key points discussed between the patient and doctor.
- Ensure each summary bullet is clear and self-contained, collectively providing a comprehensive view of the interaction.

# Output Format

Format your summary as three (and only three) bullet points, each expressing a key aspect of the conversation. Do not include any introductory, explanatory, or concluding text—only the summary bullets.

# Examples

Example 1:

- The patient reported sharp lower back pain and morning stiffness.
- The doctor indicated lumbar disc herniation or sciatica might be the cause.
- An MRI was scheduled for further diagnosis.


# Important reminders:
- Always provide three bullet points—no more, no less.
- Cover the most significant aspects or decisions in the conversation.
- Do not include patient names or any identifying information. 

'''

async def _extract_summary_for_row(
    client: AsyncOpenAIClient,
    semaphore: asyncio.Semaphore,
    text: Any,
) -> str:
    """
    Generates a summary for a single row using a local system prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(text) if text is not None else ""},
    ]

    async with semaphore:
        response = await client.generate_from_messages(messages=messages)
        return str(response.content)


async def extract_summary_dataframe(
    df: pl.DataFrame,
    model_name: str,
    *,
    source_column: str = "Dialogue",
    output_column: str = "Dialogue_summary",
    concurrency: int = 5,
    batch_size: int = 200,
    sleep_between_batches: float = 0.0,
    # max_connections: int = 50,
    # max_keepalive_connections: int = 20,
    # timeout_seconds: float = 60.0,
) -> pl.DataFrame:
    """
    Use a local system prompt + user transcript to extract a summary from a column.
    Adds a new column with the generated summaries and displays progress.
    """
    if source_column not in df.columns:
        raise ValueError(f"Missing required column: {source_column}")

    total_rows = df.shape[0]
    failed_count = 0

    client = AsyncOpenAIClient(
        model_name=model_name,
        # max_connections=max_connections,
        # max_keepalive_connections=max_keepalive_connections,
        # timeout_seconds=timeout_seconds,
    )
    semaphore = asyncio.Semaphore(concurrency)

    summaries: list[str] = []
    rows_iter: Iterable[dict[str, Any]] = df.select([source_column]).iter_rows(named=True)
    batch: list[asyncio.Task[str]] = []

    async def flush_batch() -> None:
        nonlocal failed_count
        if not batch:
            return

        results = await asyncio.gather(*batch, return_exceptions=True)
        batch.clear()

        batch_failures = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Summary extraction failed: {result}")
                summaries.append("")
                batch_failures += 1
            else:
                summaries.append(result)
        
        failed_count += batch_failures
        processed_count = len(summaries)
        
        # Write progress to the console
        progress_msg = f"Progress: {processed_count}/{total_rows} processed | Failures: {failed_count}"
        sys.stdout.write(f"\r{progress_msg.ljust(80)}")
        sys.stdout.flush()
        if sleep_between_batches > 0:
            await asyncio.sleep(sleep_between_batches)

    for row in rows_iter:
        task = asyncio.create_task(
            _extract_summary_for_row(
                client,
                semaphore,
                row.get(source_column),
            )
        )
        batch.append(task)
        if len(batch) >= batch_size:
            await flush_batch()

    # Process the final batch if any records are left
    await flush_batch()

    # Print a final newline to clear the progress bar
    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info(f"Summary extraction complete. Total failures: {failed_count}")

    return df.with_columns([pl.Series(name=output_column, values=summaries)])
