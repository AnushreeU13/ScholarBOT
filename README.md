# ScholarBOT v10: Reducing Hallucinations in Clinical RAG

ScholarBOT is a working theory and proof of concept focused on reducing hallucinations and providing verifiable citations within clinical and healthcare settings. By grounding every response in a specialized knowledge base (KB), it provides clinicians with double-summarized answers (Clinical and Patient summaries) backed by direct snippets from medical literature.

Clinicians or physicians can ask questions to ScholarBOT, which will fetch the answer from its internal knowledge base. The system responds in two distinct ways: a **Clinical Summary** for professional use and a **Patient Summary** designed for direct patient communication. All answers are accompanied by citations featuring exact words from the cited articles to ensure transparency.

---

## Key Principles

### 1. Safety-First: Abstention is a Choice
In a medical context, providing an uncertain or incorrect answer is as dangerous as providing a wrong one. ScholarBOT is designed with a **Fail-Closed** philosophy: if the knowledge base does not contain the necessary information to answer a query with high confidence, the system will consciously **Abstain** rather than synthesize a response. Nothing is "made up"—every word is grounded in the retrieved evidence.

### 2. Dual-Audience Outputs
*   **Clinical Summary**: A precise, technical abstract for medical professionals.
*   **Patient Summary**: A jargon-free explanation that translates clinical findings into accessible language for patients, without losing factual grounding.

### 3. Traceability
Every claim is mapped to a specific citation. These citations include the **exact snippets** from the source article, allowing for immediate human verification.

### 4. Flexible Knowledge Base
The system primarily answers from its pre-indexed medical guidelines and drug labels. However, users can also **upload their own documents** (PDFs) to ask specific questions about new or private datasets, which are then processed with the same strict grounding rules.

---

## How to Execute

### 1. Prerequisites
You will need an **OpenAI API Key** to enable the clinical reasoning and summarization features.

### 2. Installation
Ensure you have Python 3.9+ installed, then install the required dependencies:
```bash
pip install streamlit langchain-community PyPDF2 sentence-transformers faiss-cpu openai rank-bm25
```

### 3. Launch the Application
Navigate to the `ScholarBOT_v9` directory and run the startup script:
```bash
python start_app.py
```
*The application will automatically open in your default browser at `http://localhost:8501`.*

### 4. Provide Authorization
Once the UI loads, enter your **OpenAI API Key** in the "Access Control" section of the sidebar. You can then begin querying the system.

---

## System Architecture

*   **`app.py`**: The Streamlit web interface.
*   **`rag_pipeline_aligned.py`**: The core RAG engine implementing Hybrid Search (Dense + Sparse), RRF Merging, and the Self-Critique loop.
*   **`aligned_backend.py`**: Connects the UI to the RAG pipeline.
*   **`config.py`**: Central configuration for similarity thresholds and feature toggles.
*   **`eval/`**: Comprehensive evaluation suite using the RAGAS framework to measure faithfulness and relevancy.
