import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import mlflow

import polars as pl

from icd_codes.components.retrieval import ICDRetriever
from icd_codes.components.selection import ICDSelector
from icd_codes.components.summary import NoteSummarizer
from icd_codes.db.neon import NeonConfig, insert_predictions
from icd_codes.evaluation.evaluate import add_evaluation_details, evaluate_pipeline
from icd_codes.logger import get_logger
from icd_codes.exception import CustomException
from icd_codes.utils.config import load_config
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


@dataclass
class ICDPipelineConfig:
    input_csv: str
    output_csv: str
    limit: int
    seed: int | None
    summary_model: str
    selector_model: str
    skip_summarization: bool
    skip_rerank: bool
    retrieve_top_k: int
    rerank_top_k: int
    final_top_k: int
    reranker_type: str
    blend_alpha: float
    force_rebuild_artifacts: bool
    summary_concurrency: int = 20
    selector_concurrency: int = 10
    use_mlflow: bool = False
    mlflow_experiment: str | None = None
    mlflow_tracking_uri: str | None = None
    neon_dsn_env: str = "NEON_DB_URL"
    neon_table: str = "icd_predictions"
    neon_enabled: bool = True
    csv_enabled: bool = False


class ICDPipeline:

    def __init__(self, config: ICDPipelineConfig) -> None:
        self.config = config

    def run(self) -> None:
        """
        Main entry point for the pipeline.
        Orchestrates the setup and execution, handling MLflow context if enabled.
        """
        try:
            if self.config.use_mlflow:
                self._setup_mlflow()
                
                with mlflow.start_run():
                    self._execute_pipeline_steps(mlflow_run=True)
            else:
                logger.info("MLflow logging DISABLED. Use --mlflow to enable.")
                self._execute_pipeline_steps(mlflow_run=False)
        except Exception as exc:
            logger.exception("Pipeline failed")
            raise CustomException(exc, __import__("sys")) from exc

    def _setup_mlflow(self) -> None:
        """Configures MLflow if enabled in the config."""
        cfg = self.config
        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        else:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
        if cfg.mlflow_experiment:
            mlflow.set_experiment(cfg.mlflow_experiment)
        try:
            mlflow.openai.autolog()
        except Exception as exc:
            logger.warning(f"MLflow OpenAI autolog unavailable or failed: {exc}")
    
    def _execute_pipeline_steps(self, mlflow_run: bool) -> None:
        """
        Executes the core logical steps of the pipeline.
        """
        # 1. Load and summarize Data
        df = self._load_data()
        summarized_df, summary_traces = self._summarize_notes(df, mlflow_run)

        # 3. Retrieve & Rerank
        selection_inputs = self._retrieve_candidates(summarized_df)

        # 4. Select ICD Codes
        selections = self._select_codes(selection_inputs, mlflow_run)

        # 5. Format Output
        output_rows, selection_traces = self._format_output(selection_inputs, selections)

        # 6. Save (optional) & Evaluate
        out_df, metrics, out_path = self._save_and_evaluate(output_rows)

        # 7. Insert into Neon (optional)
        if self.config.neon_enabled:
            self._insert_neon(out_df, mlflow_run)

        # 8. Log Artifacts (if MLflow)
        if mlflow_run:
            self._log_mlflow_artifacts(out_path, metrics, summary_traces, selection_traces)

    def _load_data(self) -> pl.DataFrame:
        """Loads and preprocesses the input CSV."""
        cfg = self.config
        df = pl.read_csv(cfg.input_csv)
        df = _normalize_columns(df)

        note_col = "Note"
        if note_col not in df.columns:
            raise ValueError("Missing required column: Note")

        if cfg.limit > 0:
            df = df.sample(cfg.limit, seed=cfg.seed)

        logger.info(f"Loaded {df.height} rows from {cfg.input_csv}")
        if cfg.seed is not None and cfg.limit > 0:
            logger.info(f"Sample seed: {cfg.seed}")
        logger.info(f"Using source column for summarization: {note_col}")
        return df

    def _summarize_notes(
        self, df: pl.DataFrame, mlflow_run: bool
    ) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
        """Summarizes the notes in the DataFrame."""
        cfg = self.config
        note_col = "Note"
        notes = [str(x or "") for x in df.get_column(note_col).to_list()]

        if cfg.skip_summarization:
            summarized_df = df.with_columns([pl.Series(name="Note_summary", values=notes)])
            return summarized_df, []
        
        summarizer = NoteSummarizer(
            model_name=cfg.summary_model,
            concurrency=cfg.summary_concurrency,
        )

        if mlflow_run:

            @mlflow.trace(name="summarize_notes")
            def _summarize_traced():
                return summarizer.summarize_sync(notes)

            summaries = _summarize_traced()
        else:
            summaries = summarizer.summarize_sync(notes)

        summarized_df = df.with_columns([pl.Series(name="Note_summary", values=summaries)])
        
        summary_traces: list[dict[str, Any]] = []
        if mlflow_run:
            summary_traces = [
                {"note": n, "summary": s, "model": cfg.summary_model}
                for n, s in zip(notes, summaries)
            ]
            
        return summarized_df, summary_traces

    def _retrieve_candidates(self, summarized_df: pl.DataFrame) -> list[dict[str, Any]]:
        """Retrieves and reranks ICD candidates for each note."""
        cfg = self.config
        logger.info("Loading retrieval artifacts")
        retriever = ICDRetriever(
            retrieve_top_k=cfg.retrieve_top_k,
            rerank_top_k=cfg.rerank_top_k,
            blend_alpha=cfg.blend_alpha,
            reranker_type=cfg.reranker_type,
            force_rebuild_artifacts=cfg.force_rebuild_artifacts,
            skip_rerank=cfg.skip_rerank,
        )

        selection_inputs: list[dict[str, Any]] = []
        rows = summarized_df.iter_rows(named=True)
        note_col = "Note"

        for i, row in enumerate(rows, start=1):
            note_text = str(row.get(note_col, "") or "")
            summary_text = str(row.get("Note_summary", "") or "").strip()
            if not summary_text:
                summary_text = note_text

            reranked = retriever.retrieve_rerank(note_summary=summary_text)
            selection_inputs.append(
                {
                    "row_id": i,
                    "note_text": note_text,
                    "note_summary": summary_text,
                    "ground_truth_icd": row.get("ICD10", ""),
                    "icd_desc_ground_truth": row.get("ICD10_desc", ""),
                    "dialogue": row.get("Dialogue", ""),
                    "reranked_candidates": reranked,
                }
            )

            if i % 10 == 0 or i == summarized_df.height:
                logger.info(f"Processed {i}/{summarized_df.height}")
        
        return selection_inputs

    def _select_codes(
        self, selection_inputs: list[dict[str, Any]], mlflow_run: bool
    ) -> list[dict[str, Any]]:
        """Selects the best ICD codes using an LLM."""
        cfg = self.config
        logger.info("Selecting ICD codes via LLM")
        selector = ICDSelector(
            model_name=cfg.selector_model,
            concurrency=cfg.selector_concurrency
        )

        if mlflow_run:

            @mlflow.trace(name="select_icd_codes")
            def _select_traced():
                return asyncio.run(selector.batch_select(selection_inputs, cfg.final_top_k))

            selections = _select_traced()
        else:
            selections = asyncio.run(selector.batch_select(selection_inputs, cfg.final_top_k))
            
        return selections

    def _format_output(
        self, selection_inputs: list[dict[str, Any]], selections: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Formats the pipeline results into output rows and traces."""
        cfg = self.config
        output_rows: list[dict[str, Any]] = []
        selection_traces: list[dict[str, Any]] = []

        for row, llm_selected in zip(selection_inputs, selections):
            predictions = llm_selected.get("predictions", [])
            primary_icd = llm_selected.get("primary_icd_code", "")
            
            if self.config.use_mlflow:
                ground_truth = row.get("ground_truth_icd", "")
                is_correct = bool(ground_truth and primary_icd and primary_icd == ground_truth)
                selection_traces.append(
                    {
                        "row_id": row["row_id"],
                        "note_summary": row["note_summary"],
                        "ground_truth_icd": ground_truth,
                        "predicted_primary_icd": primary_icd,
                        "is_correct": is_correct,
                        "prompt": llm_selected.get("prompt", ""),
                        "raw_output": llm_selected.get("raw_output", ""),
                        "model": llm_selected.get("trace_model", cfg.selector_model),
                        "predictions": predictions,
                    }
                )

            output_rows.append(
                {
                    "row_id": row["row_id"],
                    "Note": row["note_text"],
                    "Note_summary": row["note_summary"],
                    "dialogue": row.get("dialogue", ""),
                    "ground_truth_icd": row.get("ground_truth_icd", ""),
                    "icd_desc_ground_truth": row.get("icd_desc_ground_truth", ""),
                    "predicted_primary_icd": primary_icd,
                    "predictions_json": json.dumps(predictions, ensure_ascii=True),
                    "reranked_candidates_json": json.dumps(row["reranked_candidates"], ensure_ascii=True),
                    "llm_raw_output_json": _safe_json_text(llm_selected.get("raw_output", "")),
                }
            )
        
        return output_rows, selection_traces

    def _save_and_evaluate(
        self, output_rows: list[dict[str, Any]]
    ) -> tuple[pl.DataFrame, dict[str, float], Path | None]:
        """Optionally saves output CSV and runs evaluation."""
        cfg = self.config
        out_df = pl.DataFrame(output_rows)
        out_df = add_evaluation_details(out_df)
        out_path: Path | None = None
        if cfg.csv_enabled:
            out_path = Path(cfg.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.write_csv(str(out_path))
            logger.info(f"Saved predictions: {out_path}")

        metrics = evaluate_pipeline(out_df)
        logger.info(f"Metrics: {metrics}")
        return out_df, metrics, out_path

    def _insert_neon(self, out_df: pl.DataFrame, mlflow_run: bool) -> None:
        cfg = self.config
        run_id = self._get_run_id(mlflow_run)
        rows = out_df.iter_rows(named=True)
        inserted = insert_predictions(
            rows,
            config=NeonConfig(
                dsn_env=cfg.neon_dsn_env,
                table=cfg.neon_table,
                run_id=run_id,
            ),
        )
        logger.info(f"Inserted {inserted} rows into Neon ({cfg.neon_table})")

    def _get_run_id(self, mlflow_run: bool) -> str:
        if mlflow_run:

            active = mlflow.active_run()
            if active:
                return active.info.run_id
        return datetime.now(timezone.utc).isoformat()

    def _log_mlflow_artifacts(
        self,
        out_path: Path | None,
        metrics: dict[str, float],
        summary_traces: list[dict[str, Any]],
        selection_traces: list[dict[str, Any]],
    ) -> None:
        """Logs parameters, metrics, and artifacts to MLflow."""
        cfg = self.config
        
        mlflow.log_params(
            {
                "input_csv": cfg.input_csv,
                "output_csv": cfg.output_csv,
                "limit": cfg.limit,
                "seed": cfg.seed,
                "summary_model": cfg.summary_model,
                "selector_model": cfg.selector_model,
                "retrieve_top_k": cfg.retrieve_top_k,
                "rerank_top_k": cfg.rerank_top_k,
                "final_top_k": cfg.final_top_k,
                "blend_alpha": cfg.blend_alpha,
                "force_rebuild_artifacts": cfg.force_rebuild_artifacts,
                "summary_concurrency": cfg.summary_concurrency,
                "selector_concurrency": cfg.selector_concurrency,
            }
        )
        mlflow.log_metrics(metrics)
        if out_path is not None:
            mlflow.log_artifact(str(out_path))
        
        out_dir = Path(cfg.output_csv).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if summary_traces:
            trace_path = out_dir / "summary_traces.jsonl"
            with trace_path.open("w", encoding="utf-8") as f:
                for item in summary_traces:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")
            mlflow.log_artifact(str(trace_path))
            
        if selection_traces:
            trace_path = out_dir / "selection_traces.jsonl"
            with trace_path.open("w", encoding="utf-8") as f:
                for item in selection_traces:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")
            mlflow.log_artifact(str(trace_path))

def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {c: c.strip() for c in df.columns if c != c.strip()}
    if rename_map:
        df = df.rename(rename_map)
    return df

def _safe_json_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=True)
    txt = str(raw).strip()
    if not txt:
        return ""
    try:
        json.loads(txt)
        return txt
    except Exception:
        return json.dumps({"raw_text": txt}, ensure_ascii=True)


def run_pipeline(
    input_csv: str,
    output_csv: str,
    limit: int,
    seed: int | None,
    summary_model: str,
    selector_model: str,
    skip_summarization: bool,
    skip_rerank: bool,
    retrieve_top_k: int,
    rerank_top_k: int,
    final_top_k: int,
    reranker_type: str,
    blend_alpha: float,
    force_rebuild_artifacts: bool,
    summary_concurrency: int = 20,
    selector_concurrency: int = 10,
    use_mlflow: bool = False,
    mlflow_experiment: str | None = None,
    mlflow_tracking_uri: str | None = None,
    neon_dsn_env: str = "NEON_DB_URL",
    neon_table: str = "icd_predictions",
    neon_enabled: bool = True,
    csv_enabled: bool = False,
) -> None:
    config = ICDPipelineConfig(
        input_csv=input_csv,
        output_csv=output_csv,
        limit=limit,
        seed=seed,
        summary_model=summary_model,
        selector_model=selector_model,
        skip_summarization=skip_summarization,
        skip_rerank=skip_rerank,
        retrieve_top_k=retrieve_top_k,
        rerank_top_k=rerank_top_k,
        final_top_k=final_top_k,
        reranker_type=reranker_type,
        blend_alpha=blend_alpha,
        force_rebuild_artifacts=force_rebuild_artifacts,
        summary_concurrency=summary_concurrency,
        selector_concurrency=selector_concurrency,
        use_mlflow=use_mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
        neon_dsn_env=neon_dsn_env,
        neon_table=neon_table,
        neon_enabled=neon_enabled,
        csv_enabled=csv_enabled,
    )
    ICDPipeline(config).run()


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ICD10 pipeline: summarize -> retrieve -> rerank -> LLM select",
        add_help=False,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = _base_parser()
    pre_args, _ = pre_parser.parse_known_args()

    config_data = {}
    if pre_args.config:
        try:
            config_data = load_config(pre_args.config)
        except FileNotFoundError:
            config_data = {}

    parser = argparse.ArgumentParser(description="ICD10 pipeline: summarize -> retrieve -> rerank -> LLM select")
    parser.set_defaults(**config_data)
    parser.add_argument("--config", default=pre_args.config, help="Path to YAML config file.")
    parser.add_argument("--input-csv", default=config_data.get("input_csv", "data/train.csv"))
    parser.add_argument("--output-csv", default=config_data.get("output_csv", "data/test/icd_predictions.csv"))
    parser.add_argument("--limit", type=int, default=config_data.get("limit", 300))
    parser.add_argument("--seed", type=int, default=config_data.get("seed", 42))
    parser.add_argument("--summary-model", default=config_data.get("summary_model", "gpt-5-mini"))
    parser.add_argument("--selector-model", default=config_data.get("selector_model", "gpt-5-mini"))
    parser.add_argument(
        "--skip-summarization",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("skip_summarization", False),
        help="Skip summarization and use raw note text directly.",
    )
    parser.add_argument(
        "--skip-rerank",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("skip_rerank", False),
        help="Skip reranking and use retrieved candidates directly.",
    )
    parser.add_argument("--retrieve-top-k", type=int, default=config_data.get("retrieve_top_k", 20))
    parser.add_argument("--rerank-top-k", type=int, default=config_data.get("rerank_top_k", 8))
    parser.add_argument("--final-top-k", type=int, default=config_data.get("final_top_k", 5))
    parser.add_argument("--reranker-type", default=config_data.get("reranker_type", "sapbert"))
    parser.add_argument("--blend-alpha", type=float, default=config_data.get("blend_alpha", 0.2))
    parser.add_argument(
        "--force-rebuild-artifacts",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("force_rebuild_artifacts", False),
    )
    parser.add_argument("--summary-concurrency", type=int, default=config_data.get("summary_concurrency", 20))
    parser.add_argument("--selector-concurrency", type=int, default=config_data.get("selector_concurrency", 10))
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("mlflow", False),
    )
    parser.add_argument("--mlflow-experiment", default=config_data.get("mlflow_experiment", "icd10-pipeline"))
    parser.add_argument("--mlflow-tracking-uri", default=config_data.get("mlflow_tracking_uri", "http://127.0.0.1:5000"))
    parser.add_argument("--neon-dsn-env", default=config_data.get("neon_dsn_env", "NEON_DB_URL"))
    parser.add_argument("--neon-table", default=config_data.get("neon_table", "icd_predictions"))
    parser.add_argument(
        "--neon",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("neon", True),
        help="Enable/disable inserting results into Neon.",
    )
    parser.add_argument(
        "--csv",
        action=argparse.BooleanOptionalAction,
        default=config_data.get("csv", False),
        help="Enable/disable writing the output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        limit=args.limit,
        seed=args.seed,
        summary_model=args.summary_model,
        selector_model=args.selector_model,
        skip_summarization=args.skip_summarization,
        skip_rerank=args.skip_rerank,
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_k=args.rerank_top_k,
        final_top_k=args.final_top_k,
        reranker_type=args.reranker_type,
        blend_alpha=args.blend_alpha,
        force_rebuild_artifacts=args.force_rebuild_artifacts,
        summary_concurrency=args.summary_concurrency,
        selector_concurrency=args.selector_concurrency,
        use_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        neon_dsn_env=args.neon_dsn_env,
        neon_table=args.neon_table,
        neon_enabled=args.neon,
        csv_enabled=args.csv,
    )


if __name__ == "__main__":
    main()
