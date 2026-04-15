# ScholarBOT v13: Clinical Assistant

ScholarBOT is a fail-closed, hallucination-resistant medical question-answering system for **Tuberculosis (TB)** and **Pneumonia (CAP)**. Every answer is grounded strictly in its knowledge base — if the evidence is insufficient, the system abstains rather than guessing. All responses include direct citations with page numbers.

---

## Core Design Principles

- **Fail-closed** — abstains when confidence is low rather than generating uncertain answers
- **Evidence-only** — no outside clinical knowledge is ever added; every sentence is sourced from retrieved documents
- **Dual-audience output** — Clinical Summary (technical, for clinicians) and Patient Summary (plain language, for patients)
- **Full traceability** — every claim links to a KB name, document, and page number
- **Context retention** — follow-up questions with pronouns ("How is it diagnosed?") are resolved against conversation history before retrieval

---

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key (`gpt-4o-mini` is used by default)
- ~4 GB disk space (for the BGE-large embedding model, downloaded automatically on first run)

---

## Installation

```bash
pip install streamlit langchain-community PyPDF2 sentence-transformers faiss-cpu openai rank-bm25 transformers torch
```

---

## Running ScholarBOT

**Step 1 — Set your OpenAI API key**

PowerShell:
```powershell
$env:OPENAI_API_KEY='sk-proj-...'
```

CMD:
```cmd
set OPENAI_API_KEY=sk-proj-...
```

**Step 2 — Navigate to the project folder**

```bash
cd path/to/ScholarBOT_v12 - Copy
```

**Step 3 — Launch the app**

```bash
python start_app.py
```

The browser will open automatically at `http://localhost:8501`.
If it doesn't, open that URL manually.

**Step 4 — Enter your API key in the sidebar**

Once the UI loads, paste your OpenAI API key in the **Access Control** field in the left sidebar. You can then start asking questions.

**To stop the server:** press `Ctrl+C` in the terminal.

---

## Using the Interface

### Asking questions
Type your clinical question in the chat box. ScholarBOT will:
1. Route the query to the appropriate knowledge base (guidelines or drug labels)
2. Retrieve and rerank relevant evidence
3. Generate a clinician answer and a patient-friendly rewrite
4. Return citations with document name and page number

If it cannot find sufficient evidence, it will respond: **No Confidence — Abstaining**.

### Search scope (sidebar)
| Option | Behaviour |
|---|---|
| All sources | Searches guidelines KB + drug labels KB (default) |
| User doc only | Searches only your uploaded PDF |
| All + User doc | Searches all KBs including your upload |

### Uploading your own documents
Use the **Upload PDF** section in the sidebar to add your own document. After uploading, switch to **User doc only** mode to ask questions specifically about that document. The same strict evidence-only rules apply.

### Multi-turn conversation
You can ask follow-up questions naturally. ScholarBOT resolves pronouns and references against the previous answer, so "How is it treated?" after a TB question will correctly retrieve TB treatment information.

---

## Knowledge Base

The pre-built knowledge base covers:
- **guidelines_kb** — TB and Pneumonia clinical guidelines and research papers
- **druglabels_kb** — Drug labels for TB and Pneumonia medications (isoniazid, rifampin, pyrazinamide, ethambutol, azithromycin, levofloxacin, etc.)

Source files are stored in `dataset/`. FAISS indices are in `faiss_indices/`.

---

## System Architecture

```
User Query
    |
    v
Context Manager (07) — coreference resolution, pronoun rewriting
    |
    v
LLM Router (08) — domain classification, KB selection, abstain signal
    |
    v
Hybrid Retrieval (09) — BGE-large dense search + BM25 sparse search per KB
    |
    v
RRF Merge + Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    |
    v
Confidence Gate — abstain if best score below threshold
    |
    v
Evidence Sufficiency Check — LLM binary YES/NO before generation
    |
    v
Answer Generation (10) — structured JSON, evidence-only, chunk-ID citations
    |
    v
Self-Critique Loop — prune any unsupported claims
    |
    v
Patient Rewrite — plain language, no new facts added
    |
    v
Response: Clinician Summary + Patient Summary + References
```

### File map

| File | Role |
|---|---|
| `01_config.py` | Central configuration — thresholds, model names, paths |
| `02_embedding_utils.py` | BGE-large-en-v1.5 embedder singleton (1024-dim) |
| `03_storage_utils.py` | FAISS index creation/loading; BM25 persistence |
| `04_pdf_utils.py` | PDF text extraction with reference-section stripping |
| `05_chunking_utils.py` | Token-aware semantic chunking (SciBERT tokenizer) |
| `06_ingest_user.py` | User PDF upload — chunk, embed, index into user_kb |
| `07_context_manager.py` | Rolling topic summary + LLM coreference resolution |
| `08_router.py` | LLM-based domain/intent/KB router with JSON output |
| `09_retriever.py` | Hybrid retrieval, RRF, cross-encoder reranking, sufficiency check |
| `10_pipeline.py` | RAG generation — Q&A, drug info, and summarization paths |
| `11_backend.py` | Engine class — wires all components, formats responses |
| `12_app.py` | Streamlit UI (port 8501) |
| `start_app.py` | Launch script |

---

## Evaluation

MIRAGE benchmark evaluation is available in `eval/mirage_eval.py`. It tests ScholarBOT against TB and Pneumonia questions from the MIRAGE medical QA benchmark using an LLM-as-judge strategy, and separately measures the abstain rate on out-of-domain questions.

```bash
# Smoke test (20 in-domain + 10 out-of-domain)
python eval/mirage_eval.py --limit 20 --ood 10

# Full evaluation (all 146 TB/Pneumonia questions + 50 out-of-domain)
python eval/mirage_eval.py

# Resume an interrupted run
python eval/mirage_eval.py --resume
```

Results are saved to `eval/eval results/mirage_results.json`.

---

## Configuration

Key settings in `01_config.py`:

| Setting | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for generation, routing, and rewriting |
| `KB_SIM_THRESHOLD` | 0.5 (guidelines/drugs), 0.3 (user) | Confidence gate — below this, system abstains |
| `TOP_K_DENSE` | 20 | Dense retrieval candidates per KB |
| `TOP_K_SPARSE` | 20 | BM25 retrieval candidates per KB |
| `RERANK_K` | 12 | Chunks kept after cross-encoder reranking |
| `CHUNK_SIZE` | 400 tokens | Size of each text chunk |
