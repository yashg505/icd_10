import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# macOS runtime guard for mixed native libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SUMMARY_MAX_LENGTH = 512
ICD_DESC_MAX_LENGTH = 64


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Candidate:
    code: str
    desc: str
    score: float | None = None


class SapBERTReranker:
    """
    Re-rank ICD candidates using SapBERT cosine similarity.

    Note: The default SapBERT model uses [CLS] pooling. If you want mean pooling,
    use cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token.

    Expected candidate format for `rerank`:
    - list[tuple[str, str]]  -> (code, desc)
    - list[tuple[str, str, float]] -> (code, desc, base_score)
    - list[dict] with keys {'code','desc'} (+ optional 'score')
    """

    def __init__(self, model_name: str = SAPBERT_MODEL_NAME):
        self.model_name = model_name
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str], max_length: int) -> np.ndarray:
        if not texts:
            hidden = self.model.config.hidden_size
            return np.empty((0, hidden), dtype=np.float32)

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)

        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy().astype("float32")

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

    def rerank(
        self,
        note_summary: str,
        candidates: list[Any],
        top_k: int | None = None,
        blend_alpha: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        blend_alpha:
        - 0.0 -> pure SapBERT rerank
        - (0,1] -> blend with candidate base score when provided:
          final = alpha * base_score + (1-alpha) * sapbert_score
        """
        if not candidates:
            return []

        parsed = self._parse_candidates(candidates)

        query = note_summary.strip()
        if not query:
            raise ValueError("note_summary must be non-empty.")

        q_vec = self.embed([query], max_length=SUMMARY_MAX_LENGTH)
        d_vec = self.embed([c.desc for c in parsed], max_length=ICD_DESC_MAX_LENGTH)

        sap_scores = (d_vec @ q_vec[0]).astype(float)

        items: list[dict[str, Any]] = []
        for cand, sap_score in zip(parsed, sap_scores):
            base = cand.score
            if base is None or blend_alpha <= 0:
                final = float(sap_score)
            else:
                final = float(blend_alpha * float(base) + (1.0 - blend_alpha) * float(sap_score))

            items.append(
                {
                    "code": cand.code,
                    "desc": cand.desc,
                    "base_score": None if base is None else float(base),
                    "sapbert_score": float(sap_score),
                    "rerank_score": float(sap_score),
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
) -> list[dict[str, Any]]:
    """Convenience function for one-shot reranking."""
    reranker = SapBERTReranker()
    return reranker.rerank(
        note_summary=note_summary,
        candidates=candidates,
        top_k=top_k,
        blend_alpha=blend_alpha,
    )


if __name__ == "__main__":
    # Demo usage with dummy candidates.
    sample_note_summary = (
        "Other specified injuries of the head initial encounter; "
        "dizziness and brief LOC after fall."
    )
    sample_candidates = [
        ("S09.8XXA", "Other specified injuries of head, initial encounter", 0.84),
        ("S06.0X1A", "Concussion with loss of consciousness of 30 minutes or less, initial encounter", 0.83),
        ("R51.9", "Headache, unspecified", 0.70),
    ]

    reranker = SapBERTReranker()
    reranked = reranker.rerank(sample_note_summary, sample_candidates, top_k=3, blend_alpha=0.2)
    for row in reranked:
        print(row)
