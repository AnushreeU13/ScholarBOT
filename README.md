# ScholarBOT: A Novel Fail-Closed Clinical RAG System

ScholarBOT is an advanced **Retrieval-Augmented Generation (RAG)** system originally built for high-stakes medical and clinical environments. It is specifically tuned for Tuberculosis and Pneumonia treatment guidelines, as well as pharmaceutical drug labels.

Unlike standard conversational AI, ScholarBOT is engineered with a strict **"Fail-Closed" philosophy**: the system prioritizes factual precision, verifiability, and safety over simple answer generation. If the system cannot find explicit, grounded evidence to answer a clinical question, it will safely "Abstain" rather than risk hallucinating a dangerous medical response.

---

## 🌟 The Novelty: What makes ScholarBOT Special?

ScholarBOT introduces several specialized, novel mechanisms that distinguish it from standard document-Q&A bots:

### 1. The "Fail-Closed" Architecture
Most LLMs are programmed to be "helpful," often leading to confident hallucinations when data is missing. ScholarBOT reverses this. It utilizes strict similarity thresholds and entailment gates. If the retrieved context confidence falls too low, or if the generated claims lack 100% evidence support during self-check, ScholarBOT actively terminates the response and returns **Abstain**.

### 2. Dual-Audience Output Pipelines
Translating complex clinical guidelines for patient consumption is notoriously difficult. ScholarBOT automatically generates two distinct, perfectly cohesive versions of every grounded answer:
*   **Clinician Summary**: A precise, grounded abstract for medical professionals containing minimal boilerplate and direct clinical evidence.
*   **Patient Summary**: A fully translated, jargon-free paragraph specifically calibrated for an **average educated person**. It automatically unpacks complex diseases into accessible explanations while strictly adhering to the clinical source text to prevent helpful hallucination drift.

### 3. Self-Critique Verification Loop
ScholarBOT includes a built-in algorithmic "peer-review". After an initial Clinician Summary is drafted by the LLM, the system essentially critiques its own work against the source evidence. It strictly prunes any ungrounded claims or hallucinated assumptions *before* outputting the text to the user.

### 4. Strict Context Locking for Private Uploads (v9 Upgrade)
When a user targets a private clinical document via the **User Document Only** UI configuration, the RAG pipeline enforces absolute semantic isolation. The backend mathematically mutes all broader medical knowledge bases, forcibly deletes supplemental citation routines, and explicitly commands both the Generative and Self-Critique LLMs to extract exclusively from the uploaded text with zero outside data contamination.

### 5. Algorithmic Fixes for PDF OCR Artifacts
Medical PDFs are often notoriously poorly formatted. ScholarBOT employs novel, active regex-cleaning algorithms within its RAG pipeline to stitch together broken text wrapping and carriage returns. This completely eliminates UI fragmentation and delivers professionally cohesive paragraphs.

### 6. Hybrid Search Engine
Combines **Dense Semantic Search** (via vector embeddings using `all-MiniLM-L6-v2`) and **Sparse Keyword Search** (BM25 matching) to perform deep, highly accurate document retrieval that catches both conceptual meaning and exact drug names.

---

## 🚀 How to Run ScholarBOT

The application features a fully integrated Streamlit web interface.

### 1. Zero-Configuration Startup
Because the necessary OpenAI Authentication keys have been dynamically embedded natively into the local repository framework to automatically bypass external blocks, there is absolutely **no manual environment setup required**.

### 2. Launch the Web Application
Simply navigate to the project directory in your terminal and execute the main runner script. This will concurrently spin up the Python backend components and automatically trigger your default web browser to pop open directly into the interface:
```bash
python start_app.py
```
*The app automatically bounds to `http://localhost:8501`.*

### 3. Exposing via REST API
For developer integration, ScholarBOT includes a fully headless REST API layer via FastAPI (`api.py`).
1. Install FastAPI and Uvicorn: `pip install fastapi uvicorn pydantic`
2. Start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
3. Test the endpoint:
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the adverse hepatic reactions for Isoniazid?", "user_uploaded_available": false}'
```

---

## 📂 System Architecture Breakdown

### ⚡ Entry Points
*   **`start_app.py`**: The primary runner that initializes the server.
*   **`app.py`**: The Streamlit frontend housing the UI, chat interface, and upload handlers.

### 🧠 Core RAG Engine
*   **`rag_pipeline_aligned.py`**: The "Brain" of ScholarBOT. This file orchestrates the Hybrid Search, Reranking, OCR text-stitching, Self-Critique loops, and patient translation execution.
*   **`router.py`**: A specialized intent classifier that routes queries optimally between general guidelines, drug-label data, and user uploads.
*   **`aligned_backend.py`**: The API service layer connecting the Streamlit frontend to the complex RAG operations.

### ⚙️ Utilities & Configuration
*   **`config.py`**: The central control file for thresholds, feature toggles, and file paths.
*   **`embedding_utils.py` & `chunking_utils.py`**: Tools for processing raw text into embeddable chunks. The internal v9 architecture now executes a `MedCPTDualEmbedder` **Singleton Pattern** down through the FAISS indexing routines, functionally preventing aggressive PyTorch memory leaks when repeatedly Hot-Reloading the inference server.
*   **`user_ingest_aligned.py`**: The script responsible for rapidly ingesting, chunking, and indexing user-uploaded PDFs into temporary, isolated Vector Stores.

### 📊 Evaluation Suite (`/eval/`)
*   **`/eval/RAGAS/ragas_eval_v9.py`**: Harnesses the RAGAS framework to compute native hallucination constraint scores against our dedicated QA datasets.
*   **`/eval/NLP/`**: Isolated directory housing token precision and semantic ROUGE-L similarity testing logic.
*   **`/eval/eval results/`**: Directory natively housing the serialized evaluation output arrays and source dataset inputs.

---

## 🧪 Evaluation & Performance

ScholarBOT is rigorously evaluated using standard LLM metrics (RAGAS framework) against a curated dataset of physician-grade medical questions. 

**Current Baseline Metrics:**
*   **Faithfulness Score (Grounding)**: ~0.86+ (indicating extremely low hallucination rates).
*   **Answer Relevancy**: ~0.75+ (indicating direct, helpful responses).
*   *Note: Detailed comprehensive evaluations can be generated by running `python eval/RAGAS/ragas_eval_v9.py` from the root directory.*
