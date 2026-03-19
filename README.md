# ScholarBOT: A Fail-Closed Clinical RAG System

ScholarBOT is an advanced **Retrieval-Augmented Generation (RAG)** system built specifically for high-stakes medical and clinical environments. It is engineered with a "Fail-Closed" philosophy: the system prioritizes factual precision and safety over simple answer generation. If the system cannot find explicit, grounded evidence to answer a clinical question, it will safely "Abstain" rather than risk hallucination.

The system is particularly tuned for domains such as Tuberculosis and Pneumonia treatment guidelines, as well as pharmaceutical drug labels.

---

## 🌟 Key Features

*   **Fail-Closed Architecture**: Utilizes strict similarity and entailment gates. If retrieved context confidence falls below a set threshold (e.g., 0.35) or generated claims lack 100% evidence support, ScholarBOT triggers a safe exit (Abstain).
*   **Hybrid Search Engine**: Combines Dense (Semantic vector embeddings via `all-MiniLM-L6-v2`) and Sparse (BM25 Keyword matching) retrieval to perform deep, highly accurate document searches.
*   **Dual-Audience Summaries**: Automatically generates two versions of its findings:
    *   **Clinician Summary**: A precise, grounded abstract with bullet points for medical professionals.
    *   **Patient Summary**: A translated, cohesive summary written at a 6th-grade reading level.
*   **Strict Context Locking**: When a user uploads a private clinical document, the system automatically "mutes" the broader medical knowledge bases, ensuring answers are derived *exclusively* from the user's uploaded context.
*   **Self-Critique Verification Loop**: Includes a built-in algorithmic "peer-review". After an initial answer is drafted, the system critiques its own work against the source evidence and prunes any ungrounded claims before output.
*   **Source Citations & Observability**: Every generated claim is mapped back to the specific retrieved chunk and source document.

---

## 🚀 How to Run ScholarBOT

The application features a fully integrated Streamlit web interface.

### 1. Environment Setup
Ensure you have your OpenAI API key set in your environment variables, as this powers the generation logic.
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Live Demo Link (Temporary API Endpoint)
If you want to test the product live without installing it locally, use the temporary testing URL (via Serveo tunnel):
*   **🔗 Public Test Link**: [https://fcc946645d53611c-76-191-28-39.serveousercontent.com](https://fcc946645d53611c-76-191-28-39.serveousercontent.com)
*(Note: As this is a live tunnel from a local machine, the link will expire when the session is closed).*

### 3. Launch the Web Application
Navigate to the project directory and run the start script. This will orchestrate the Streamlit server and automatically open your default browser.
```bash
python start_app.py
```
*The app connects to `http://localhost:8501`.*

### 4. Exposing via REST API
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

### 5. Usage Flow
1. **Scope Selection**: Use the left sidebar to select your search scope (e.g., *Main KB Only*, *User Document Only*).
2. **Document Upload**: You can drag-and-drop clinical PDFs (like patient records or unique guidelines) directly into the UI.
3. **Query**: Ask your specific clinical question in the chat box (e.g., *"What are the hepatic adverse reactions for Isoniazid?"*).
4. **Review**: The system will display the Clinician Summary, Patient Summary, and the exact Evidence Context it retrieved.

---

## 📂 System Architecture Breakdown

### ⚡ Entry Points
*   **`start_app.py`**: The primary runner that initializes the server.
*   **`app.py`**: The Streamlit frontend housing the UI, chat interface, and upload handlers.

### 🧠 Core RAG Engine
*   **`rag_pipeline_aligned.py`**: The "Brain" of ScholarBOT. This file orchestrates the Hybrid Search, Reranking, Self-Critique loops, and prompt execution.
*   **`router.py`**: A specialized intent classifier that routes queries optimally between general guidelines, drug-label data, and user uploads.
*   **`aligned_backend.py`**: The API service layer connecting the Streamlit frontend to the complex RAG operations.

### ⚙️ Utilities & Configuration
*   **`config.py`**: The central control file for thresholds, feature toggles (like Strict Mode or Zero Hallucination Mode), and file paths.
*   **`embedding_utils.py` & `chunking_utils.py`**: Tools for processing raw text into embeddable chunks.
*   **`user_ingest_aligned.py`**: The script responsible for rapidly ingesting, chunking, and indexing user-uploaded PDFs into temporary, isolated Vector Stores.

---

## 🧪 Evaluation & Performance

ScholarBOT is rigorously evaluated using standard LLM metrics (RAGAS framework) against a curated dataset of physician-grade medical questions. 

**Current Baseline Metrics:**
*   **Faithfulness Score (Grounding)**: ~0.86+ (indicating extremely low hallucination rates).
*   **Answer Relevancy**: ~0.75+ (indicating direct, helpful responses).
*   *Note: Detailed comprehensive evaluations can be generated by running `python ragas_eval_v9.py` against the test set.*

---

## 🏛️ Project Design Principles
1.  **Grounding over Creativity**: In medical settings, an incorrect answer is dangerous. "I don't know" (Abstain) is the preferred safe state.
2.  **Transparent Verifiability**: Users must be able to click through to the source chunk that generated the claim.
3.  **Readability**: Outputs must avoid fragmented, broken text, ensuring cohesive sentences for both clinicians and patients.
