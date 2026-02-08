import asyncio
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from icd_codes.components.retrieval import ICDRetriever
from icd_codes.components.selection import ICDSelector
from icd_codes.components.summary import NoteSummarizer
from icd_codes.utils.config import load_config

load_dotenv()


st.set_page_config(page_title="ICD-10 Coding Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_retriever(
    retrieve_top_k: int,
    rerank_top_k: int,
    blend_alpha: float,
    force_rebuild: bool,
) -> ICDRetriever:
    return ICDRetriever(
        retrieve_top_k=retrieve_top_k,
        rerank_top_k=rerank_top_k,
        blend_alpha=blend_alpha,
        force_rebuild_artifacts=force_rebuild,
    )


def _summarize(note_text: str, model_name: str, concurrency: int = 1) -> str:
    summarizer = NoteSummarizer(model_name=model_name, concurrency=concurrency)
    return summarizer.summarize_sync([note_text])[0]


def _select_codes(
    note_summary: str,
    reranked_candidates: list[dict[str, Any]],
    model_name: str,
    final_top_k: int,
    concurrency: int = 1,
) -> dict[str, Any]:
    selector = ICDSelector(model_name=model_name, concurrency=concurrency)
    selections = asyncio.run(
        selector.batch_select(
            rows=[
                {"note_summary": note_summary, "reranked_candidates": reranked_candidates}
            ],
            final_top_k=final_top_k,
        )
    )
    return selections[0]


st.title("ICD-10 Coding Assistant")
st.caption("Single-note demo with retrieval + rerank + LLM selection.")

config = load_config("config/default.yaml")
summary_model_default = config.get("summary_model", "gpt-5-mini")
selector_model_default = config.get("selector_model", "gpt-5-mini")
summary_model_options = config.get("summary_model_options") or [summary_model_default]
selector_model_options = config.get("selector_model_options") or [selector_model_default]
if isinstance(summary_model_options, str):
    summary_model_options = [summary_model_options]
if isinstance(selector_model_options, str):
    selector_model_options = [selector_model_options]

with st.sidebar:
    st.header("Models")
    summary_model = st.selectbox(
        "Summary model",
        options=summary_model_options,
        index=summary_model_options.index(summary_model_default)
        if summary_model_default in summary_model_options
        else 0,
    )
    selector_model = st.selectbox(
        "Selector model",
        options=selector_model_options,
        index=selector_model_options.index(selector_model_default)
        if selector_model_default in selector_model_options
        else 0,
    )

    st.header("Retrieval")
    retrieve_top_k = st.number_input("Retrieve top-k", min_value=1, max_value=100, value=20)
    rerank_top_k = st.number_input("Rerank top-k", min_value=1, max_value=50, value=8)
    final_top_k = st.number_input("LLM top-k", min_value=1, max_value=10, value=5)
    blend_alpha = st.slider("Blend alpha", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    force_rebuild = st.checkbox("Force rebuild artifacts", value=False)

    st.header("Display")
    show_debug = st.checkbox("Show debug panels", value=False)
    skip_summarization = st.checkbox("Skip summarization", value=False)
    skip_rerank = st.checkbox("Skip rerank", value=False)

note = st.text_area("Clinical Note", height=220, placeholder="Paste the clinical note here...")
dialogue = st.text_area("Dialogue (optional)", height=140, placeholder="Optional dialogue text...")

run = st.button("Run ICD-10 Prediction", type="primary")

if run:
    if not note.strip() and not dialogue.strip():
        st.error("Please provide a note or dialogue.")
        st.stop()

    combined_text = note.strip()
    if dialogue.strip():
        combined_text = f"{combined_text}\n\nDialogue:\n{dialogue.strip()}" if combined_text else dialogue.strip()

    with st.status("Running pipeline...", expanded=False) as status:
        try:
            if skip_summarization:
                summary = combined_text
            else:
                status.update(label="Summarizing note...")
                summary = _summarize(combined_text, summary_model)

            status.update(label="Retrieving candidates...")
            retriever = _get_retriever(retrieve_top_k, rerank_top_k, blend_alpha, force_rebuild)
            retriever.skip_rerank = skip_rerank
            reranked = retriever.retrieve_rerank(note_summary=summary)

            status.update(label="Selecting ICD codes with LLM...")
            selection = _select_codes(
                note_summary=summary,
                reranked_candidates=reranked,
                model_name=selector_model,
                final_top_k=final_top_k,
            )
            status.update(label="Done", state="complete")
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)
            st.stop()

    st.subheader("Prediction")
    primary = selection.get("primary_icd_code", "")
    st.write(f"Primary ICD-10: `{primary or 'N/A'}`")

    preds = selection.get("predictions", [])
    if preds:
        pred_df = pd.DataFrame(preds)
        st.dataframe(pred_df, use_container_width=True)
    else:
        st.info("No predictions returned.")

    st.subheader("Summary")
    st.write(summary)

    if show_debug:
        st.subheader("Reranked Candidates")
        if reranked:
            cand_df = pd.DataFrame(reranked)
            st.dataframe(cand_df, use_container_width=True)
        else:
            st.info("No candidates returned from retriever.")

        st.subheader("Raw LLM Output")
        st.code(selection.get("raw_output", "") or "", language="json")
