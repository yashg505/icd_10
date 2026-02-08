import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ICDEvaluationConfig:
    ground_truth_col: str = "ground_truth_icd"
    predicted_col: str = "predicted_primary_icd"
    predictions_col: str = "predictions_json"
    reranked_col: str = "reranked_candidates_json"
    ground_truth_split_pattern: str | None = r"[;,|]"


@dataclass(frozen=True)
class ICDMetrics:
    total_samples: int
    accuracy_primary: float
    llm_topk_recall: float
    rerank_recall: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "accuracy_primary": self.accuracy_primary,
            "llm_topk_recall": self.llm_topk_recall,
            "rerank_recall": self.rerank_recall,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


class ICDDataParser:
    def __init__(self, config: ICDEvaluationConfig) -> None:
        self.config = config

    def to_rows(self, df: Any) -> list[dict[str, Any]]:
        if hasattr(df, "iter_rows"):
            return list(df.iter_rows(named=True))
        if hasattr(df, "to_dict"):
            try:
                return df.to_dict("records")
            except Exception:
                pass
        if isinstance(df, list):
            return df
        raise TypeError("Unsupported df type. Provide a polars/pandas DataFrame or list[dict].")

    @staticmethod
    def parse_json_list(raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            txt = raw.strip()
            if not txt:
                return []
            try:
                data = json.loads(txt)
                return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    @staticmethod
    def normalize_code(code: Any) -> str:
        if code is None:
            return ""
        return str(code).strip().upper()

    @staticmethod
    def split_codes(code: str, pattern: str | None) -> list[str]:
        if not code:
            return []
        if pattern is None:
            return [code]
        parts = re.split(pattern, code)
        return [p.strip().upper() for p in parts if p.strip()]

    @classmethod
    def safe_get_codes_from_list(cls, items: list[dict[str, Any]], key: str) -> list[str]:
        codes = []
        for item in items:
            if isinstance(item, dict) and key in item:
                codes.append(cls.normalize_code(item.get(key)))
        return [c for c in codes if c]


class ICDEvaluator:
    def __init__(self, config: ICDEvaluationConfig) -> None:
        self.config = config
        self.parser = ICDDataParser(config)

    def evaluate(self, df: Any) -> ICDMetrics:
        cfg = self.config
        rows = self.parser.to_rows(df)

        total = 0
        correct_primary = 0
        in_llm_topk = 0
        in_rerank = 0

        y_true: list[str] = []
        y_pred: list[str] = []

        for row in rows:
            gt_raw = self.parser.normalize_code(row.get(cfg.ground_truth_col))
            gt_codes = self.parser.split_codes(gt_raw, cfg.ground_truth_split_pattern)
            gt = gt_codes[0] if gt_codes else ""

            pred = self.parser.normalize_code(row.get(cfg.predicted_col))

            preds_list = self.parser.parse_json_list(row.get(cfg.predictions_col))
            preds_codes = self.parser.safe_get_codes_from_list(preds_list, "icd_code")

            reranked_list = self.parser.parse_json_list(row.get(cfg.reranked_col))
            reranked_codes = self.parser.safe_get_codes_from_list(reranked_list, "code")

            if not gt:
                continue

            total += 1
            y_true.append(gt)
            y_pred.append(pred)

            if pred and pred == gt:
                correct_primary += 1

            if gt in preds_codes:
                in_llm_topk += 1

            if gt in reranked_codes:
                in_rerank += 1

        accuracy_primary = (correct_primary / total) if total else 0.0
        llm_topk_recall = (in_llm_topk / total) if total else 0.0
        rerank_recall = (in_rerank / total) if total else 0.0

        classes = sorted(set(y_true) | set(y_pred))
        tp = Counter()
        fp = Counter()
        fn = Counter()

        for t, p in zip(y_true, y_pred):
            if not p:
                fn[t] += 1
                continue
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        def _prf(c: str) -> tuple[float, float, float]:
            tpc = tp[c]
            fpc = fp[c]
            fnc = fn[c]
            precision = tpc / (tpc + fpc) if (tpc + fpc) else 0.0
            recall = tpc / (tpc + fnc) if (tpc + fnc) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            return precision, recall, f1

        if classes:
            precisions, recalls, f1s = zip(*[_prf(c) for c in classes])
            macro_precision = sum(precisions) / len(precisions)
            macro_recall = sum(recalls) / len(recalls)
            macro_f1 = sum(f1s) / len(f1s)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        return ICDMetrics(
            total_samples=total,
            accuracy_primary=accuracy_primary,
            llm_topk_recall=llm_topk_recall,
            rerank_recall=rerank_recall,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
        )

    def add_details(self, df: Any):
        cfg = self.config
        rows = self.parser.to_rows(df)

        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            gt_raw = self.parser.normalize_code(row.get(cfg.ground_truth_col))
            gt_codes = self.parser.split_codes(gt_raw, cfg.ground_truth_split_pattern)
            gt = gt_codes[0] if gt_codes else ""

            pred = self.parser.normalize_code(row.get(cfg.predicted_col))

            preds_list = self.parser.parse_json_list(row.get(cfg.predictions_col))
            preds_codes = self.parser.safe_get_codes_from_list(preds_list, "icd_code")

            reranked_list = self.parser.parse_json_list(row.get(cfg.reranked_col))
            reranked_codes = self.parser.safe_get_codes_from_list(reranked_list, "code")

            gt_in_rerank = bool(gt and gt in reranked_codes)
            gt_rerank_rank = (reranked_codes.index(gt) + 1) if gt_in_rerank else None
            gt_in_llm = bool(gt and gt in preds_codes)

            enriched = dict(row)
            enriched["is_correct_primary"] = bool(gt and pred and pred == gt)
            enriched["gt_in_rerank_topk"] = gt_in_rerank
            enriched["gt_rerank_rank"] = gt_rerank_rank
            enriched["gt_in_llm_topk"] = gt_in_llm
            enriched_rows.append(enriched)

        if hasattr(df, "with_columns"):
            try:
                import polars as pl

                return pl.DataFrame(enriched_rows)
            except Exception:
                pass
        return enriched_rows


def evaluate_pipeline(
    df: Any,
    *,
    ground_truth_col: str = "ground_truth_icd",
    predicted_col: str = "predicted_primary_icd",
    predictions_col: str = "predictions_json",
    reranked_col: str = "reranked_candidates_json",
    ground_truth_split_pattern: str | None = r"[;,|]",
) -> dict[str, Any]:
    """
    Evaluate pipeline output.

    Metrics:
    - accuracy_primary: exact match of predicted_primary_icd vs ground_truth_icd
    - llm_topk_recall: ground_truth present in predictions_json list
    - rerank_recall: ground_truth present in reranked_candidates_json list
    - macro_precision/recall/f1 on primary prediction
    """
    cfg = ICDEvaluationConfig(
        ground_truth_col=ground_truth_col,
        predicted_col=predicted_col,
        predictions_col=predictions_col,
        reranked_col=reranked_col,
        ground_truth_split_pattern=ground_truth_split_pattern,
    )
    evaluator = ICDEvaluator(cfg)
    return evaluator.evaluate(df).to_dict()


def add_evaluation_details(
    df: Any,
    *,
    ground_truth_col: str = "ground_truth_icd",
    predicted_col: str = "predicted_primary_icd",
    predictions_col: str = "predictions_json",
    reranked_col: str = "reranked_candidates_json",
    ground_truth_split_pattern: str | None = r"[;,|]",
):
    """
    Add per-row diagnostics:
    - is_correct_primary
    - gt_in_rerank_topk
    - gt_rerank_rank (1-based rank, None if absent)
    - gt_in_llm_topk
    """
    cfg = ICDEvaluationConfig(
        ground_truth_col=ground_truth_col,
        predicted_col=predicted_col,
        predictions_col=predictions_col,
        reranked_col=reranked_col,
        ground_truth_split_pattern=ground_truth_split_pattern,
    )
    evaluator = ICDEvaluator(cfg)
    return evaluator.add_details(df)
