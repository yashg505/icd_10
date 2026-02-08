import asyncio
from typing import Any

from openai import AsyncOpenAI

from icd_codes.models.prediction import ICDSelection


def _build_selection_prompt(
    note_summary: str,
    candidates: list[dict[str, Any]],
    max_codes: int,
) -> str:
    cand_lines = []
    for i, c in enumerate(candidates, start=1):
        rerank_score = c.get("rerank_score")
        if rerank_score is None:
            rerank_score = c.get("sapbert_score")
        if rerank_score is None:
            rerank_score = c.get("medcpt_score")
        cand_lines.append(
            f"{i}. code={c['code']} | desc={c['desc']} | "
            f"retrieval_score={c.get('base_score')} | rerank_score={rerank_score} | final_score={c.get('final_score')}"
        )

    return (
        f'''
        You are an ICD-10 coding documentation interpreter.

        Your role is NOT to act as a doctor or make clinical judgments.
        Your job is to map documented text to the most appropriate ICD-10 codes.

        RULES:
        - Select codes ONLY from the candidate list.
        - Do NOT invent diagnoses.
        - Do NOT infer conditions that are not explicitly documented.
        - If an ICD code is explicitly written in the note and present in the candidate list, prefer it.
        - Prefer:
        - confirmed diagnoses > suspected conditions > symptoms
        - If wording includes "suspected", "likely", "possible", or "rule out":
        - avoid definitive disease codes unless clearly confirmed.
        - If wording includes "history of", "follow-up", or past condition:
        - consider history/encounter codes.
        - If no confirmed diagnosis is documented:
        - prefer symptom or unspecified codes.
        - Use staging, severity, and laterality only if explicitly stated.

        OUTPUT REQUIREMENTS:
        - Return at most {max_codes} predictions.
        - Each prediction must include:
        - icd_code
        - confidence (0 to 1)
        - short justification based only on the note text.

        NOTE SUMMARY (entity-dense for context):
        {note_summary}

        CANDIDATE ICD CODES:
        '''
        + "\n".join(cand_lines)
    )


class ICDSelector:
    def __init__(self, model_name: str, concurrency: int = 10) -> None:
        self.model_name = model_name
        self.concurrency = concurrency
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(concurrency)

    async def select_one(
        self,
        note_summary: str,
        reranked_candidates: list[dict[str, Any]],
        final_top_k: int,
    ) -> dict[str, Any]:
        prompt = _build_selection_prompt(
            note_summary=note_summary,
            candidates=reranked_candidates,
            max_codes=final_top_k,
        )

        async with self.semaphore:
            response = await self.client.responses.parse(
                model=self.model_name,
                input=[{"role": "user", "content": prompt}],
                text_format=ICDSelection,
            )

        parsed: ICDSelection = response.output_parsed
        if not parsed:
            best = reranked_candidates[0] if reranked_candidates else None
            return {
                "primary_icd_code": best["code"] if best else "",
                "predictions": [
                    {
                        "icd_code": best["code"],
                        "confidence": float(best.get("final_score", 0.0)),
                        "justification": "Fallback to top reranked candidate due to empty LLM output.",
                    }
                ]
                if best
                else [],
                "raw_output": "",
                "prompt": prompt,
                "trace_model": self.model_name,
            }

        data = parsed.model_dump()
        data["raw_output"] = getattr(response, "output_text", "")
        data["prompt"] = prompt
        data["trace_model"] = self.model_name
        return data

    async def batch_select(
        self,
        rows: list[dict[str, Any]],
        final_top_k: int,
    ) -> list[dict[str, Any]]:
        async def _wrap(idx: int, coro):
            return idx, await coro

        tasks = []
        for idx, row in enumerate(rows):
            coro = self.select_one(
                note_summary=row["note_summary"],
                reranked_candidates=row["reranked_candidates"],
                final_top_k=final_top_k,
            )
            tasks.append(asyncio.create_task(_wrap(idx, coro)))

        results: list[dict[str, Any]] = [None] * len(tasks)
        total = len(tasks)
        completed = 0

        for task in asyncio.as_completed(tasks):
            idx, result = await task
            results[idx] = result
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"LLM selection progress: {completed}/{total}")

        return results
