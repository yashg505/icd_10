from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable
from dotenv import load_dotenv
 
import psycopg
from psycopg.types.json import Json

load_dotenv()

@dataclass(frozen=True)
class NeonConfig:
    dsn_env: str = "NEON_DB_URL"
    table: str = "icd_predictions"
    run_id: str | None = None


def _json_or_none(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None


def _get_dsn(env_key: str) -> str:
    dsn = os.getenv(env_key, "").strip()
    if not dsn:
        raise ValueError(f"Missing database DSN in env var: {env_key}")
    return dsn


def insert_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    config: NeonConfig,
) -> int:
    dsn = _get_dsn(config.dsn_env)
    inserted = 0

    sql = f"""
        insert into {config.table} (
            run_id,
            row_id,
            note,
            summarized_note,
            dialogue,
            icd_ground_truth,
            icd_desc_ground_truth,
            predicted_icd_code,
            is_correct,
            gt_in_rerank_topk,
            gt_rerank_rank,
            predictions,
            reranked_candidates,
            llm_raw_output
        )
        values (
            %(run_id)s,
            %(row_id)s,
            %(note)s,
            %(summarized_note)s,
            %(dialogue)s,
            %(icd_ground_truth)s,
            %(icd_desc_ground_truth)s,
            %(predicted_icd_code)s,
            %(is_correct)s,
            %(gt_in_rerank_topk)s,
            %(gt_rerank_rank)s,
            %(predictions)s,
            %(reranked_candidates)s,
            %(llm_raw_output)s
        )
        on conflict (run_id, row_id) do update set
            note = excluded.note,
            summarized_note = excluded.summarized_note,
            dialogue = excluded.dialogue,
            icd_ground_truth = excluded.icd_ground_truth,
            icd_desc_ground_truth = excluded.icd_desc_ground_truth,
            predicted_icd_code = excluded.predicted_icd_code,
            is_correct = excluded.is_correct,
            gt_in_rerank_topk = excluded.gt_in_rerank_topk,
            gt_rerank_rank = excluded.gt_rerank_rank,
            predictions = excluded.predictions,
            reranked_candidates = excluded.reranked_candidates,
            llm_raw_output = excluded.llm_raw_output
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                predictions = _json_or_none(row.get("predictions_json") or row.get("predictions"))
                reranked_candidates = _json_or_none(
                    row.get("reranked_candidates_json") or row.get("reranked_candidates")
                )
                llm_raw_output = _json_or_none(
                    row.get("llm_raw_output_json") or row.get("llm_raw_output")
                )

                payload = {
                    "run_id": config.run_id or row.get("run_id") or "unknown",
                    "row_id": row.get("row_id"),
                    "note": row.get("Note") or row.get("note"),
                    "summarized_note": row.get("Note_summary") or row.get("summarized_note"),
                    "dialogue": row.get("dialogue"),
                    "icd_ground_truth": row.get("icd_ground_truth") or row.get("ground_truth_icd"),
                    "icd_desc_ground_truth": row.get("icd_desc_ground_truth"),
                    "predicted_icd_code": row.get("predicted_primary_icd") or row.get("predicted_icd_code"),
                    "is_correct": row.get("is_correct")
                    if "is_correct" in row
                    else row.get("is_correct_primary"),
                    "gt_in_rerank_topk": row.get("gt_in_rerank_topk"),
                    "gt_rerank_rank": row.get("gt_rerank_rank"),
                    "predictions": Json(predictions) if predictions is not None else None,
                    "reranked_candidates": Json(reranked_candidates)
                    if reranked_candidates is not None
                    else None,
                    "llm_raw_output": Json(llm_raw_output) if llm_raw_output is not None else None,
                }
                cur.execute(sql, payload)
                inserted += 1
        conn.commit()

    return inserted
