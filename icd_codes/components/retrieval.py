from dataclasses import dataclass
from typing import Any

from icd_codes.components.create_embedding_openai import (
    load_or_create_icd_artifacts,
    retrieve_icd_for_note_summary_with_artifacts,
)
from icd_codes.components.rerank_medcpt import MedCPTCrossEncoderReranker
from icd_codes.components.rerank_sapbert import SapBERTReranker


@dataclass
class ICDRetriever:
    retrieve_top_k: int
    rerank_top_k: int
    blend_alpha: float
    reranker_type: str = "sapbert"
    force_rebuild_artifacts: bool = False
    skip_rerank: bool = False

    def __post_init__(self) -> None:
        self.index, self.code_map = load_or_create_icd_artifacts(
            force_rebuild=self.force_rebuild_artifacts
        )
        reranker_key = (self.reranker_type or "sapbert").strip().lower()
        if reranker_key == "sapbert":
            self.reranker = SapBERTReranker()
        elif reranker_key in {"medcpt", "medcpt-cross-encoder", "cross-encoder"}:
            self.reranker = MedCPTCrossEncoderReranker()
        else:
            raise ValueError(f"Unknown reranker_type: {self.reranker_type}")

    def retrieve_rerank(self, note_summary: str) -> list[dict[str, Any]]:
        retrieved = retrieve_icd_for_note_summary_with_artifacts(
            note_summary=note_summary,
            index=self.index,
            code_map=self.code_map,
            top_k=self.retrieve_top_k,
        )
        if self.skip_rerank:
            sliced = retrieved[: self.rerank_top_k] if self.rerank_top_k and self.rerank_top_k > 0 else retrieved
            normalized: list[dict[str, Any]] = []
            for item in sliced:
                if isinstance(item, dict):
                    normalized.append(item)
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    code, desc, score = item[0], item[1], item[2]
                    normalized.append(
                        {
                            "code": code,
                            "desc": desc,
                            "base_score": score,
                            "sapbert_score": None,
                            "rerank_score": None,
                            "final_score": score,
                        }
                    )
                else:
                    normalized.append(
                        {
                            "code": str(item),
                            "desc": "",
                            "base_score": None,
                            "sapbert_score": None,
                            "rerank_score": None,
                            "final_score": None,
                        }
                    )
            return normalized
        reranked = self.reranker.rerank(
            note_summary=note_summary,
            candidates=retrieved,
            top_k=self.rerank_top_k,
            blend_alpha=self.blend_alpha,
        )
        return reranked
