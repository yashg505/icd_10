import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
ICD_DESC_MAX_LENGTH = 64
SUMMARY_MAX_LENGTH = 512
BATCH_SIZE = 128
MAX_RETRIES = 3

ICD_DATA_PATH = Path("data/medsynth.csv")
ARTIFACT_DIR = Path("icd_codes/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = ARTIFACT_DIR / "icd_index_openai.faiss"
CODE_MAP_PATH = ARTIFACT_DIR / "code_map_openai.pkl"

# Optional demo-only source, used under __main__.
SUMMARY_DEMO_PATH = Path("data/test/medsynth_summarized_notes.csv")

_client = None


def resolve_col(df: pd.DataFrame, options: list[str], required: bool = True) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lowered:
            return lowered[opt.lower()]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {options}. Found: {list(df.columns)}")
    return None


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def normalize_np(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return emb / norms


def embed(texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = get_openai_client()
    all_vecs: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                all_vecs.append(vecs)
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(1.5 * attempt)

    emb = np.vstack(all_vecs).astype(np.float32)
    return normalize_np(emb)


def build_icd_artifacts(icd_data_path: Path = ICD_DATA_PATH) -> tuple[faiss.Index, dict[int, tuple[str, str]]]:
    print(f"Loading ICD source: {icd_data_path}")
    icd_src = pd.read_csv(icd_data_path)
    icd_src.columns = [c.strip() for c in icd_src.columns]

    icd_col = resolve_col(icd_src, ["ICD10"])
    desc_col = resolve_col(icd_src, ["ICD10_desc"])

    icd_src = icd_src.dropna(subset=[icd_col, desc_col]).copy()
    icd_src[icd_col] = icd_src[icd_col].astype(str)
    icd_src[desc_col] = icd_src[desc_col].astype(str).map(lambda x: truncate_words(x, ICD_DESC_MAX_LENGTH))
    icd_df = icd_src[[icd_col, desc_col]].drop_duplicates().reset_index(drop=True)

    print(f"Building OpenAI ICD artifacts from {len(icd_df)} unique descriptions...")
    icd_np = embed(icd_df[desc_col].tolist())

    index = faiss.IndexFlatIP(icd_np.shape[1])
    index.add(icd_np)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved: {INDEX_PATH}")

    code_map = {i: (row[icd_col], row[desc_col]) for i, row in icd_df.iterrows()}
    with open(CODE_MAP_PATH, "wb") as f:
        pickle.dump(code_map, f)
    print(f"Saved: {CODE_MAP_PATH}")

    return index, code_map


def load_or_create_icd_artifacts(force_rebuild: bool = False) -> tuple[faiss.Index, dict[int, tuple[str, str]]]:
    if not force_rebuild and INDEX_PATH.exists() and CODE_MAP_PATH.exists():
        print(f"Using existing FAISS index: {INDEX_PATH}")
        print(f"Using existing code map: {CODE_MAP_PATH}")
        index = faiss.read_index(str(INDEX_PATH))
        with open(CODE_MAP_PATH, "rb") as f:
            code_map = pickle.load(f)
        return index, code_map

    return build_icd_artifacts(ICD_DATA_PATH)


def retrieve_icd_for_note_summary(
    note_summary: str,
    top_k: int = 5,
    force_rebuild: bool = False,
) -> list[tuple[str, str, float]]:
    index, code_map = load_or_create_icd_artifacts(force_rebuild=force_rebuild)
    return retrieve_icd_for_note_summary_with_artifacts(
        note_summary=note_summary,
        index=index,
        code_map=code_map,
        top_k=top_k,
    )


def retrieve_icd_for_note_summary_with_artifacts(
    note_summary: str,
    index: faiss.Index,
    code_map: dict[int, tuple[str, str]],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    if index.ntotal == 0:
        return []

    k = min(max(1, top_k), index.ntotal)
    query = truncate_words(str(note_summary), SUMMARY_MAX_LENGTH)
    q_np = embed([query])

    distances, ids = index.search(q_np, k)

    results = []
    for score, idx in zip(distances[0], ids[0]):
        if idx < 0:
            continue
        code, desc = code_map[idx]
        results.append((code, desc, float(score)))
    return results


if __name__ == "__main__":
    if SUMMARY_DEMO_PATH.exists():
        demo_df = pd.read_csv(SUMMARY_DEMO_PATH)
        demo_df.columns = [c.strip() for c in demo_df.columns]
        summary_col = resolve_col(demo_df, ["note_summary", "Note_summary"], required=False)

        if summary_col is not None and len(demo_df) > 0:
            sample_query = str(demo_df.iloc[0][summary_col])
        else:
            sample_query = "Other specified injuries of the head initial encounter"
    else:
        sample_query = "Other specified injuries of the head initial encounter"

    print("\nDemo query:")
    print(sample_query)
    for r in retrieve_icd_for_note_summary(sample_query, top_k=5):
        print(r)
