"""
12_app.py
Streamlit web interface for ScholarBOT v13.
Architecture position: 12 — the user-facing layer.

Run with:
    streamlit run 12_app.py
or via start_app.py.
"""

import os
import shutil
import importlib
from pathlib import Path

import streamlit as st

_cfg = importlib.import_module("01_config")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarBOT: Clinical Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages          = []
    st.session_state.uploaded_doc_name = None
    st.session_state.needs_kb_reload   = False

    # Security purge: clear any leftover user files from a previous session
    try:
        for path in [
            _cfg.PROJECT_ROOT / "temp_uploads",
            _cfg.FAISS_INDICES_DIR / _cfg.KB_USER,
            _cfg.KB_PROCESSED_DIR / "user_fact",
        ]:
            if Path(path).exists():
                shutil.rmtree(path, ignore_errors=True)
        print("[ScholarBOT] Session purge complete.")
    except Exception as e:
        print(f"[ScholarBOT] Purge warning: {e}")

# ── Engine (cached per API key) ───────────────────────────────────────────────
@st.cache_resource
def load_engine(api_key: str):
    if not api_key:
        return None
    _backend = importlib.import_module("11_backend")
    return _backend.ScholarBotEngine(api_key=api_key, verbose=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # API key
    st.subheader("Access Control")
    api_key_input = st.text_input(
        "OpenAI API Key", type="password",
        help="Required to enable clinical reasoning.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.session_state.api_key = api_key_input
    elif st.session_state.get("api_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key

    if not st.session_state.get("api_key"):
        st.warning("Please provide an OpenAI API key to begin.")

    st.markdown("---")

    # File upload
    st.subheader("Upload Document (optional)")
    uploaded_file = st.file_uploader("Upload a PDF to query against", type=["pdf"])

    if uploaded_file:
        temp_dir  = _cfg.PROJECT_ROOT / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.session_state.uploaded_doc_name != uploaded_file.name:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    _ingest = importlib.import_module("06_ingest_user")
                    stats   = _ingest.ingest_user_pdf(file_path, doc_name=uploaded_file.name)

                    added = stats.get("added_chunks", 0)
                    if added == 0:
                        st.error("Zero chunks extracted. Is this a scanned (image-only) PDF?")
                    else:
                        st.success(f"Ingested {added} chunks from {uploaded_file.name}.")
                        st.caption(f"Pages: {stats.get('num_pages')} | "
                                   f"Characters: {stats.get('total_chars'):,}")

                    st.session_state.uploaded_doc_name = uploaded_file.name
                    st.session_state.needs_kb_reload   = True

                except Exception as e:
                    import traceback
                    st.error(f"Ingestion failed: {e}")
                    st.code(traceback.format_exc())

    # Search scope
    st.markdown("---")
    st.subheader("Search Scope")
    search_mode = st.radio(
        "Where should ScholarBOT look?",
        ["Standard (Guidelines + Drug Labels)", "User Document Only"],
        index=1 if st.session_state.uploaded_doc_name else 0,
        disabled=not st.session_state.uploaded_doc_name,
        help="'User Document Only' searches only your uploaded PDF.",
    )

    st.markdown("---")

    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        engine = load_engine(st.session_state.get("api_key", ""))
        if engine:
            engine.reset_context()
        st.rerun()

    st.markdown("---")
    st.markdown("""
**Fail-Closed Design**

ScholarBOT abstains rather than guesses.
If evidence is insufficient or confidence is low, it will say so.
This is a safety feature, not a limitation.
""")

# ── Load engine ───────────────────────────────────────────────────────────────
engine = None
try:
    engine = load_engine(st.session_state.get("api_key", ""))
    if engine and st.session_state.get("needs_kb_reload"):
        engine.reload_user_kb()
        st.session_state.needs_kb_reload = False
except Exception as e:
    st.error(f"Failed to load ScholarBOT engine: {e}")
    st.stop()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("ScholarBOT: Clinical Assistant")
st.caption(
    f"**Scope:** `{search_mode}` | "
    f"**Domain:** Tuberculosis & Pneumonia (CAP) | "
    f"**Mode:** Fail-Closed (evidence-only)"
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if not st.session_state.get("api_key"):
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

if not engine:
    st.warning("Engine not loaded yet. Please wait or refresh.")
    st.stop()

if prompt := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    force_user = (search_mode == "User Document Only")

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Searching evidence..."):
            try:
                response_text, confidence, meta = engine.generate_response(
                    query=prompt,
                    force_user_kb=force_user,
                    history=st.session_state.messages[:-1],
                )
            except Exception as e:
                response_text = f"An error occurred: {e}"
                confidence    = 0.0
                meta          = {"status": "error", "evidence_chunks": []}

        placeholder.markdown(response_text)

        # Evidence expander
        with st.expander("View Retrieved Evidence & Metadata"):
            st.write(f"**Confidence Score:** `{confidence:.3f}`")
            st.write(f"**Status:** `{meta.get('status', 'unknown')}`")
            st.write(f"**Source KB:** `{meta.get('source', 'N/A')}`")

            route = meta.get("route", {})
            if route:
                st.write(f"**Domain:** `{route.get('domain', 'N/A')}` | "
                         f"**Intent:** `{route.get('intent', 'N/A')}`")

            chunks = meta.get("evidence_chunks", [])
            if chunks:
                st.subheader("Source Chunks")
                for c in chunks:
                    st.markdown(
                        f"**[{c.get('chunk_id', '?')}]** "
                        f"*{c.get('citation', 'Unknown source')}*"
                    )
                    st.caption((c.get("text") or "")[:400] + "...")
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
