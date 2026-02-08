# ICD-10 Prediction System

## Overview
This project builds a system that takes clinical dialog/notes and predicts the most relevant **ICD-10 code**. The repository implements a modular, inspectable pipeline so each stage can be evaluated independently (summarization → retrieval → reranking → final selection) that makes it easier to debug accuracy issues and iterate on components without retraining everything end-to-end.

At a high level, the workflow is:
1. Ingest notes and normalize columns.
2. Summarize long/low-signal notes into a compact clinical summary.
3. Retrieve candidate ICD codes using embeddings + vector search.
4. Rerank the candidates with a medical-specific model.
5. Select the final code with an LLM.
6. Evaluate and log metrics/outputs.

## Problem Statement
Build a system that can take a dialog/notes and predict the correct ICD-10 code.

### Data Snapshot
- ~10,000 data points
- \>2,000 distinct ICD codes
- ~5 samples per ICD code on average
- Multi-class classification problem (one primary ICD code per note)

### Success Metrics
- **Overall accuracy**
- **Top-k accuracy** (e.g., Top-3 / Top-5)

## Approaches Considered
We explored multiple strategies. This repository implements the **RAG-style pipeline** , and keeps the others documented as alternatives.

1. **Multi-agent system (alternative)**
   - Agents: doctor, ICD coding practitioner, patient, insurance provider.
   - They debate and converge on a final ICD code.
   - Pros: richer reasoning, explainability.
   - Cons: expensive, slow, harder to operationalize.

2. **RAG-style pipeline (implemented)**
   - Summarize → retrieve → rerank → LLM selection.
   - Modular components so accuracy can be tested at each stage.

3. **Similarity matching (alternative)**
   - Map notes to an embedding space, find nearest ICD code based on similarity.
   - When a new note arrives, return the nearest code by distance.
   - Pros: fast and simple.
   - Cons: weak reasoning and fragile with small-per-class data.

### How current approach could be improved:
Based on experiments, I found following gaps:
1. Retrieval process and re-ranker process is working at 84% accuracy, while LLM working at 71% accuracy.
2. Gap is present in LLM final selection and retieval and re-ranking process. 
3. LLM final selection gap may be minimized by as understood by analyzing the results:
    - improving the prompt
    - passing better summarized notes(or passing without summarizing the notes)


4. Retrieval-reranker process can be improved by(further gains maybe be incremental here):
    - checking with different embedding/cross encoder models.
    - summarizing the notes in different way, which doesn't conceal important information.
    - We are using openai embeddings, which we can replace with a stronger models which are trained on medical data.


## Implemented Architecture

**Core pipeline:** `icd_codes/pipeline.py`

## System Flow
1. **Load data**: Read input CSV and normalize required columns.
2. **Summarize notes**: Generate concise clinical summaries (optional skip).
3. **Embed ICD codes**: Build embeddings for ICD code descriptions and store in FAISS.
4. **Embed & retrieve**: Embed summaries and retrieve top-k ICD candidates with FAISS.
5. **Rerank**: Use a medical reranker (SapBERT/MedCPT) to refine candidates.
6. **Select**: LLM selects the final ICD code using the summary + candidates and provides justification.
7. **Evaluate**: Compute accuracy, F1, and top-k metrics.
8. **Log**: Store outputs in MLflow/Neon (optional).

## Architecture Diagram (Mermaid)
```mermaid
flowchart LR
    A[Input Notes / Dialog] --> B[Data Ingestion]
    B --> C[LLM Summarizer]
    C --> D[Summary Embedding]
    K[ICD Code Descriptions] --> L[ICD Embedding Index - FAISS]
    D --> M[Retrieve Top-k Candidates]
    L --> M
    M --> E[Reranker<br/>SapBERT / MedCPT]
    E --> F[LLM Selector<br/>Inputs: Summary + Candidates]
    C --> F
    subgraph J["MLflow / Neon DB"]
        G[Predicted ICD Code]
        H[Justification / Explanation]
        I[Evaluation Metrics]
    end
    F --> G
    F --> H
    F --> I
```

**Key components:**
- `icd_codes/components/data_ingestion.py`
  - Load CSV data and standardize required columns.
- `icd_codes/components/summary.py`
  - LLM-based summarization to reduce noise and token length.
- `icd_codes/components/create_embedding_openai.py`
  - Builds embeddings for ICD descriptions and stores a FAISS index.
- `icd_codes/components/retrieval.py`
  - Retrieves the top-k candidate codes using vector similarity.
- `icd_codes/components/rerank_medcpt.py`
  - Medical-domain reranker using MedCPT.
- `icd_codes/components/rerank_sapbert.py`
  - Medical-domain reranker using SapBERT.
- `icd_codes/components/selection.py`
  - LLM-based final selection and justification.
- `icd_codes/evaluation/evaluate.py`
  - Computes accuracy, F1, and top-k metrics.

**Observability and logging:**
- `icd_codes/db/neon.py` stores prediction traces (optional).
- `mlruns/` and `mlflow.db` store experiment logs.

## Tools & Tech
- **Python** + **Polars** for data processing
- **OpenAI Embeddings** + **FAISS** for retrieval
- **SapBERT / MedCPT** for reranking
- **OpenAI LLM** for summarization and final selection
- **MLflow** for experiment tracking
- **Neon (Postgres)** for optional logging

## Installation

Prerequisites:
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

```bash
# Clone the repository
git clone <repo_url>
cd icd10

# Install dependencies using uv
uv sync

# OR using pip (if prefered)
pip install .
```

## Configuration

1. Create a `.env` file in the root directory.
2. Add the following environment variables:

```bash
OPENAI_API_KEY="openai_api_key"
MLFLOW_TRACKING_URI="sqlite:///mlruns.db"  # or  MLflow tracking URI
NEON_DB_URL="postgresql://user:password@host/dbname..." # Optional: for Neon DB logging
```

## Getting Started

### Prerequisites
- Python 3.13+
- `uv` (recommended) or `pip`

### Installation
```bash
# Clone the repository
git clone https://github.com/yashg505/icd_10.git
cd icd10

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Environment Configuration
This project uses environment variables loaded from a `.env` file at the repo root.

```bash
# .env (example)
OPENAI_API_KEY=sk-...
NEON_DB_URL=postgres://...
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Runtime parameters live in `config/default.yaml` and can be overridden via CLI flags.

## Usage
```bash
# Default run
python main.py

# Custom run
python main.py --config config/production.yaml --limit 100 --mlflow
```

### Streamlit App
To launch the interactive UI:
```bash
uv run streamlit run apps/streamlit_app.py
```

### Expanded Usage Examples
```bash

# run to get the data
uv run icd_codes/components/data_ingestion.py

# Run with custom config and save results to CSV (disable Neon inserts)
uv run main.py --config config/production.yaml --csv --no-neon

# Run a quick test on a small sample without summarization
uv run main.py --limit 10 --skip-summarization
```

## Evaluation
Metrics are logged via MLflow and also exported as artifacts. Key metrics:
- Accuracy
- Top-k accuracy (in re-ranked candidates)
- f1-score

## Future Work
- **FastAPI service** for real-time inference.
- **Deterministic Approach** - Stack in rule based approches along with LLM reasoning
- **Cloud Run deployment** Containerize and deploy on cloud run(GCP) for scalable serving.
- **Embedding caching** for faster response times.
- **Alternate approach evaluation**: compare multi-agent pipeline and pure similarity baseline.
