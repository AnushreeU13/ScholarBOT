# ScholarBOT v4: Tiered Clinical RAG Architecture

ScholarBOT v4 is a **Fail-Closed Retrieval Augmented Generation (RAG)** system designed for high-stakes clinical question answering. It prioritizes factual precision and safety over exhaustive recall.

## 🚀 How to Run

The application includes a single startup script that handles the server and browser launch.

### 1. Start the Application
Open your terminal in the `ScholarBOT_v4` directory and run:

```bash
python start_app.py
```

This command will:
1.  Start the Streamlit server on port **8501**.
2.  Automatically launch your default browser (Chrome recommended) to `http://localhost:8501`.

### 2. Upload & Query
1.  In the browser, use the sidebar to **upload a PDF document** (e.g., specific treatment guidelines).
2.  Select **"User Document Only"** as the Search Scope.
3.  Ask your clinical question (e.g., *"What are the treatment outcomes for TB?"*).

---

## 📂 File Descriptions

### Entry Points
- **`start_app.py`**: The primary entry point. Orchestrates the Streamlit server launch and opens the browser window.
- **`app.py`**: The main Streamlit frontend application. Handles UI layout, file uploads, and chat interaction.

### Core Pipeline
- **`rag_pipeline_aligned.py`**: The central RAG engine. Implements the retrieval logic, similarity gating, "Fail-Closed" mechanisms, and answer generation.
- **`aligned_backend.py`**: The service layer that connects the frontend (`app.py`) to the RAG pipeline. Handles session state and response formatting.
- **`router.py`**: Intent classification module. Routes user queries to the appropriate knowledge base (Guidelines, Drugs, or User Uploads).

### Configuration & Data
- **`config.py`**: Central configuration file. Defines file paths, similarity thresholds (0.40), model settings, and prompts.
- **`user_ingest_aligned.py`**: Handles the ingestion of user-uploaded PDFs. Extracts text, chunks it, and builds a temporary FAISS index (`user_kb`).

### Utilities
- **`pdf_utils.py`**: Functions for extracting text from PDF files using `pdfplumber` or `pypdf`.
- **`chunking_utils.py`**: Logic for splitting text into semantic chunks with overlap.
- **`embedding_utils.py`**: Wrapper for the embedding model (`sentence-transformers/all-MiniLM-L6-v2`).
- **`storage_utils.py`**: specific functions for saving/loading FAISS indices.
- **`deduplication_utils.py`**: Helpers to remove duplicate content during ingestion.
- **`llm_utils.py`**: Interfaces with the LLM (Local or API) for generating answers.

### Documentation
- **`Methodology_Paper_Ready.md`**: A detailed technical report outlining the tiered architecture, "Fail-Closed" design, and evaluation metrics.

---

## 🏛️ Architecture Highlights

### 1. Fail-Closed Design
If the retrieval system cannot find explicit evidence in the guidelines matching your question (similarity < 0.40), it will **Abstain** rather than hallucinate.

### 2. Tiered Retrieval
- **Index Segregation**: Separate FAISS indices for Guidelines, Drug Labels, and User Data.
- **Section Boosting**: Heuristically boosts critical sections (e.g., "Adverse Reactions") to ensure relevant context.

### 3. Entailment Verification
An internal gate checks if the generated answer is logically supported by the retrieved context. If not, the system returns "No confidence".
