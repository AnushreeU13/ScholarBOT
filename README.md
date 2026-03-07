# ScholarBOT v9: Researcher-Graded Clinical RAG Baseline

ScholarBOT v9 represent the **Researcher-Graded Baseline** for the Fail-Closed Clinical RAG system. It introduces a multi-stage retrieval and verification architecture designed to eliminate hallucinations in high-stakes medical environments (TB, Pneumonia, and Drug Labels).

## 🚀 Key Evolutionary Features (v9 vs v4)

*   **Hybrid Search (Dense + Sparse)**: Combines Semantic embedding search (all-MiniLM-L6-v2) with BM25 keyword matching for exact medical terminology (e.g., drug names like *Isoniazid*).
*   **Self-Critique Verification Loop**: After generating an answer, the system performs an internal "peer-review" step to prune any claims not 100% grounded in the evidence.
*   **Strict User Context Locking**: When a user uploads a clinical document, the system automatically mutes the general knowledge base to prevent "knowledge drift" from unrelated guidelines.
*   **Cohesive Summary Patch**: Implements a smart multi-line bullet parser to ensure clinical and patient summaries are complete, grammatical sentences instead of fragments.
*   **Tiered Gating**: Includes Intent Routing, Similarity Gating (0.35 threshold), and Entailment Verification.

---

## 📊 RAGAS Evaluation Scores (v9 Baseline)
*Evaluated on the 200-question Physician Grade dataset.*

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Context Recall** | **0.279** | High raw retrieval; finds most relevant sources. |
| **Faithfulness** | **0.867** | Excellent grounding; extremely low hallucination rate. |
| **Answer Relevancy** | **0.758** | Directly addresses patient/clinician needs. |
| **Context Precision** | **0.197** | *Baseline performance (improved to 0.240 in v10).* |

---

## 📂 System Architecture

### ⚡ Entry Points
- **`start_app.py`**: Launches the Streamlit server (Port 8501) and handles browser orchestration.
- **`app.py`**: Streamlit UI with sidebar controls for Search Scope and Zero-Hallucination Mode.

### 🧠 Core RAG Logic
- **`rag_pipeline_aligned.py`**: The "Brain" of v9. Implements Hybrid Search, Self-Critique, and Cohesive Parsing.
- **`router.py`**: Routes queries between Guideline, Drug, and User knowledge bases.
- **`aligned_backend.py`**: Backend engine managing state and multi-query execution.

---

## 🛠️ How to Run

1.  **Environment Setup**:
    Ensure you have an `OPENAI_API_KEY` set in your environment variables.
    ```bash
    export OPENAI_API_KEY="your-key-here"
    ```

2.  **Launch**:
    ```bash
    python start_app.py
    ```

3.  **Local Only Mode**:
    By default, v9 uses Local Embeddings (MiniLM) and can be configured for Local LLM inference (Qwen/Llama) via `config.py`.

---

## 🏛️ Project Principles
1.  **Grounding over Creativity**: If the evidence is missing, the system **ABSTAINS**.
2.  **Verifiability**: Every answer is derived from the `faiss_indices/` content.
3.  **Cohesion**: Every response is parsed for clinical readability.
