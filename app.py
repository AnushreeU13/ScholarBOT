
import streamlit as st
import time
import os
import shutil
from pathlib import Path
from aligned_backend import AlignedScholarBotEngine
from user_ingest_aligned import ingest_user_pdf

# Page Config
st.set_page_config(page_title="ScholarBOT v11", page_icon="🤖", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Security Feature: Purge user files on new session / browser restart
    try:
        from config import PROJECT_ROOT, FAISS_INDICES_DIR, DATA_DIR
        temp_dir = PROJECT_ROOT / "temp_uploads"
        user_faiss = FAISS_INDICES_DIR / "user_kb"
        user_chunks = DATA_DIR / "user_fact"
        
        if temp_dir.exists(): shutil.rmtree(temp_dir, ignore_errors=True)
        if user_faiss.exists(): shutil.rmtree(user_faiss, ignore_errors=True)
        if user_chunks.exists(): shutil.rmtree(user_chunks, ignore_errors=True)
        print("[ScholarBOT] Local security purge completed. Cleared previous user documents.")
    except Exception as e:
        print(f"[ScholarBOT] Purge error: {e}")

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")
    
    # 0. API KEY (REQUIRED)
    st.subheader("Access Control")
    api_key_input = st.text_input("Enter OpenAI API Key", type="password", help="ScholarBOT requires a valid OpenAI API key.")
    
    # Ultra-safe environment setting
    if api_key_input and isinstance(api_key_input, str):
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.session_state.openai_api_key = api_key_input
    elif st.session_state.get("openai_api_key") and isinstance(st.session_state.get("openai_api_key"), str):
         os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    if not st.session_state.get("openai_api_key"):
        st.warning("**OpenAI API Key Required**: Please provide your key above to enable clinical reasoning.")
    
    # 1. File Upload
    st.subheader("Upload User Document")
    uploaded_file = st.file_uploader("Upload a PDF to query against", type=["pdf"])
    
    if uploaded_file:
        # Save temp file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Ingest if new
        if st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    stats = ingest_user_pdf(file_path, doc_name=uploaded_file.name)
                    st.success(f"Ingestion Complete!")
                    st.json({
                        "Chunks Added": stats.get('added_chunks', 0),
                        "Total Characters": stats.get('total_chars', 0),
                        "Pages Processed": stats.get('num_pages', 0),
                        "Debug Sample": stats.get('debug_sample', 'N/A')
                    })
                    if stats.get('added_chunks', 0) == 0:
                        st.error("⚠️ Zero chunks extracted! Is this a scanned PDF (image-only)? Try a text-based PDF.")

                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.needs_reload = True
                except Exception as e:
                    import traceback
                    st.error(f"Ingestion failed: {e}")
                    st.code(traceback.format_exc())

    # 2. Search Mode
    search_mode = st.radio(
        "Search Scope:",
        ["Standard (Guidelines + Drugs)", "User Document Only"],
        index=1 if st.session_state.uploaded_file_name else 0,
        disabled=not st.session_state.uploaded_file_name
    )
    
    # 3. Actions
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Note on 'No Confidence':**
    ScholarBOT v9 uses a **Fail-Closed** design. 
    If the retrieval system cannot find explicit evidence in the guidelines matching your question, it will **Abstain** rather than hallucinate.
    
    This is a safety feature, not a bug.
    """)

# Initialize Engine (Cached)
@st.cache_resource
def load_engine(api_key):
    # Only initialize if we have a key
    if not api_key:
        return None
    return AlignedScholarBotEngine(api_key=api_key, verbose=True)

try:
    engine = load_engine(st.session_state.get("openai_api_key"))
    if st.session_state.get("needs_reload", False):
        engine.reload_user_kb()
        st.session_state.needs_reload = False
except Exception as e:
    st.error(f"Failed to load ScholarBOT engine: {e}")
    st.stop()

# Main Chat Interface
st.title("ScholarBOT v11: Clinical Assistant")
st.markdown(f"**Current Scope:** `{search_mode}`")

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if not st.session_state.get("openai_api_key"):
    st.info("Once you provide a valid API key in the sidebar, you can start asking clinical questions.")
    prompt = None
elif prompt := st.chat_input("Ask a clinical question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine Routing
    force_user = (search_mode == "User Document Only")

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing guidelines & evidences..."):
            try:
                # Call Backend — pass all prior messages (excluding the one just added)
                response_text, confidence, meta = engine.generate_response(
                    prompt,
                    force_user_kb=force_user,
                    history=st.session_state.messages[:-1],
                )

                # Process the text for streaming effect
                # Note: response_text might contain markdown like "### Clinician Summary"
                
                # FIX: Ensure bullet points have newlines for proper Markdown rendering
                # Removed aggressive bullet replacement for v11
                
                # We'll stream it character by character or word by word
                
                # Check for "Abstain" or "No confidence" to clarify UI
                is_abstain = "No confidence" in response_text
                
                if is_abstain:
                    full_response = "**Fail-Closed Triggered**\n\n" + response_text
                else:
                    full_response = response_text

                message_placeholder.markdown(full_response)
                
                # Show Metadata (Sources)
                with st.expander("View Retrieved Evidence & Metadata"):
                    # Removed raw JSON for clean presentation
                    st.write(f"**Confidence Score:** `{confidence:.2f}`")
                    st.write(f"**Source Index:** `{meta.get('source', 'Unknown')}`")
                    
                    if "evidence_chunks" in meta:
                        st.subheader("Source Chunks")
                        for i, chunk in enumerate(meta["evidence_chunks"]):
                            st.markdown(f"**Chunk {i+1}** (Source: `{chunk.get('source', 'Unknown')}`)")
                            st.caption(chunk.get('text', '')[:400] + "...")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error generating response: {e}")
                full_response = "I encountered an error processing your request."
                message_placeholder.markdown(full_response)

    # Add Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
