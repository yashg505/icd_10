import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# macOS runtime guard for mixed native libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MEDCPT_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
MAX_PAIR_LENGTH = 512


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Candidate:
    code: str
    desc: str
    score: float | None = None


class MedCPTCrossEncoderReranker:
    """
    Re-rank ICD candidates using MedCPT cross-encoder relevance scores.

    Expected candidate format for `rerank`:
    - list[tuple[str, str]]  -> (code, desc)
    - list[tuple[str, str, float]] -> (code, desc, base_score)
    - list[dict] with keys {'code','desc'} (+ optional 'score')
    """

    def __init__(self, model_name: str = MEDCPT_MODEL_NAME):
        self.model_name = model_name
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _parse_candidates(self, candidates: list[Any]) -> list[Candidate]:
        parsed: list[Candidate] = []

        for item in candidates:
            if isinstance(item, dict):
                code = item.get("code")
                desc = item.get("desc")
                if code is None or desc is None:
                    raise ValueError("Dict candidates must include 'code' and 'desc'.")
                parsed.append(Candidate(code=str(code), desc=str(desc), score=item.get("score")))
                continue

            if isinstance(item, tuple) and len(item) == 2:
                code, desc = item
                parsed.append(Candidate(code=str(code), desc=str(desc), score=None))
                continue

            if isinstance(item, tuple) and len(item) >= 3:
                code, desc, score = item[0], item[1], item[2]
                parsed.append(Candidate(code=str(code), desc=str(desc), score=float(score)))
                continue

            raise ValueError(
                "Unsupported candidate format. Use dict {'code','desc','score?'} "
                "or tuple (code, desc[, score])."
            )

        return parsed

    def _score_pairs(self, query: str, candidates: list[Candidate], max_length: int, batch_size: int) -> np.ndarray:
        scores: list[float] = []
        texts = [c.desc for c in candidates]

        for start in range(0, len(texts), batch_size):
            batch_desc = texts[start : start + batch_size]
            tokens = self.tokenizer(
                [query] * len(batch_desc),
                batch_desc,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**tokens)

            logits = out.logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.squeeze(-1)

            scores.extend(batch_scores.detach().cpu().float().tolist())

        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        note_summary: str,
        candidates: list[Any],
        top_k: int | None = None,
        blend_alpha: float = 0.0,
        max_length: int = MAX_PAIR_LENGTH,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """
        blend_alpha:
        - 0.0 -> pure MedCPT rerank
        - (0,1] -> blend with candidate base score when provided:
          final = alpha * base_score + (1-alpha) * medcpt_score
        """
        if not candidates:
            return []

        parsed = self._parse_candidates(candidates)

        query = note_summary.strip()
        if not query:
            raise ValueError("note_summary must be non-empty.")

        med_scores = self._score_pairs(query, parsed, max_length=max_length, batch_size=batch_size)

        items: list[dict[str, Any]] = []
        for cand, med_score in zip(parsed, med_scores):
            base = cand.score
            if base is None or blend_alpha <= 0:
                final = float(med_score)
            else:
                final = float(blend_alpha * float(base) + (1.0 - blend_alpha) * float(med_score))

            items.append(
                {
                    "code": cand.code,
                    "desc": cand.desc,
                    "base_score": None if base is None else float(base),
                    "medcpt_score": float(med_score),
                    "rerank_score": float(med_score),
                    "final_score": final,
                }
            )

        items.sort(key=lambda x: x["final_score"], reverse=True)
        if top_k is not None:
            items = items[: max(1, top_k)]
        return items


def rerank_icd_candidates(
    note_summary: str,
    candidates: list[Any],
    top_k: int | None = None,
    blend_alpha: float = 0.0,
    max_length: int = MAX_PAIR_LENGTH,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Convenience function for one-shot reranking."""
    reranker = MedCPTCrossEncoderReranker()
    return reranker.rerank(
        note_summary=note_summary,
        candidates=candidates,
        top_k=top_k,
        blend_alpha=blend_alpha,
        max_length=max_length,
        batch_size=batch_size,
    )
