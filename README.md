# ScholarBOT v12: Reducing Hallucinations in Clinical RAG

ScholarBOT is a working theory and proof of concept focused on reducing hallucinations and providing verifiable citations within clinical and healthcare settings. By grounding every response in a specialized knowledge base (KB), it provides clinicians with double-summarized answers (Clinical and Patient summaries) backed by direct snippets from medical literature.

Clinicians or physicians can ask questions to ScholarBOT, which will fetch the answer from its internal knowledge base. The system responds in two distinct ways: a **Clinical Summary** for professional use and a **Patient Summary** designed for direct patient communication. All answers are accompanied by citations featuring exact words from the cited articles to ensure transparency.

The current knowledge base covers **Tuberculosis (TB)** and **Pneumonia (CAP)**, indexed from raw PDF and XML source files into two FAISS indices: `guidelines_kb` and `druglabels_kb`.

---

## Key Principles

### 1. Safety-First: Abstention is a Choice
In a medical context, providing an uncertain or incorrect answer is as dangerous as providing a wrong one. ScholarBOT is designed with a **Fail-Closed** philosophy: if the knowledge base does not contain the necessary information to answer a query with high confidence, the system will consciously **Abstain** rather than synthesize a response. Nothing is "made up" — every word is grounded in the retrieved evidence.

### 2. Strict Evidence-Only Generation
All generation paths (clinician answer, self-critique loop, patient rewrite) are restricted to the retrieved evidence. The system will not supplement answers with outside clinical knowledge. If the evidence is insufficient, the output is `ABSTAIN`.

### 3. Dual-Audience Outputs
- **Clinical Summary**: A precise, technical abstract for medical professionals.
- **Patient Summary**: A jargon-free explanation that translates clinical findings into accessible language for patients, without losing factual grounding.

### 4. Traceability
Every claim is mapped to a specific citation — KB name, document name, and page number — allowing for immediate human verification.

### 5. Flexible Knowledge Base
The system primarily answers from its pre-indexed medical guidelines and drug labels. Users can also **upload their own PDFs** and query against them in isolation ("User Document Only" mode) or combined with the existing KB ("Standard" mode), processed under the same strict grounding rules.

### 6. Context Retention
Follow-up questions with pronouns or references (e.g. *"How is it diagnosed?"* after asking about TB) are automatically rewritten into self-contained clinical queries before routing and retrieval, so multi-turn conversations work correctly.

---

## How to Execute

### 1. Prerequisites
You will need an **OpenAI API Key** to enable the clinical reasoning and summarization features.

### 2. Installation
Ensure you have Python 3.9+ installed, then install the required dependencies:
```bash
pip install streamlit langchain-community PyPDF2 sentence-transformers faiss-cpu openai rank-bm25 transformers torch
```

### 3. Launch the Application
Navigate to the project directory and run the startup script:
```bash
python start_app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

Alternatively:
```bash
streamlit run app.py
```

### 4. Provide Authorization
Once the UI loads, enter your **OpenAI API Key** in the "Access Control" section of the sidebar. You can then begin querying the system.

### 5. Stop the Server
```bash
Ctrl+C
```
To clear Streamlit's resource cache:
```bash
streamlit cache clear
```

---

## System Architecture

| File | Role |
|---|---|
| `app.py` | Streamlit web interface — handles user input, session state, file uploads, and response rendering |
| `aligned_backend.py` | Engine layer — query rewriting, history wiring, claim-to-snippet alignment, response formatting |
| `rag_pipeline_aligned.py` | Core RAG engine — hybrid search (dense + sparse), RRF merging, cross-encoder reranking, multi-gate filtering, self-critique loop |
| `router.py` | Domain router — classifies queries as TB/pneumonia/drug/out-of-domain and selects target KBs |
| `config.py` | Central configuration — similarity thresholds, feature toggles, generation settings |
| `embedding_utils.py` | Embedder wrapper — BAAI/bge-large-en-v1.5 (1024-dim) via SentenceTransformer |
| `user_ingest_aligned.py` | On-demand PDF ingestion — chunks, embeds, and indexes user-uploaded documents into `user_kb` |
| `chunking_utils.py` | Token-aware semantic chunking using SciBERT tokenizer |
| `storage_utils.py` | FAISS index creation and loading |
| `eval/` | Evaluation suite using the RAGAS framework to measure faithfulness and relevancy |

---

## Pipeline Flow

```
User Query
    |
    v
Query Rewriting (coreference resolution via conversation history)
    |
    v
Domain Router (TB / Pneumonia / Drug / Out-of-domain / Abstain)
    |
    v
Hybrid Retrieval (Dense BGE-large + Sparse BM25) per target KB
    |
    v
RRF Merge + Section Bias
    |
    v
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    |
    v
Confidence Gate (threshold check -> Abstain if low)
    |
    v
Context Consolidation (merge adjacent chunks)
    |
    v
Clinician Answer Generation (evidence-only, LLM)
    |
    v
Self-Critique Loop (prune unsupported claims)
    |
    v
Patient Answer Rewrite
    |
    v
Patient Safety Gate + Consistency Check
    |
    v
Response (Clinician Summary + Patient Summary + References)
```

---

## v12 Changes

- **Context retention**: query rewriting resolves pronouns and references using conversation history before routing
- **Strict evidence-only generation**: removed permission to supplement answers with outside clinical knowledge across all generation and critique prompts
- **Meaningful confidence gate**: raised `KB_SIM_THRESHOLD` from `0.01` to `0.5` (guidelines/druglabels) and `0.3` (user uploads) so the abstain mechanism is actually enforced
- **Fixed stale page number bug**: uploaded document chunks were all being labelled with the last page's number; each chunk now correctly records its own page
- **Removed broken dead code**: `ingest_user_file` in `aligned_backend.py` had wrong tuple unpacking and non-existent FAISS API calls
- **Embedder singleton**: BAAI/bge-large-en-v1.5 (1.3 GB) is loaded once per session instead of on every upload
- **Tokenizer cache**: SciBERT tokenizer is loaded once instead of once per page during ingestion
