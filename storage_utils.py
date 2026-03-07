import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from embedding_utils import MedCPTDualEmbedder

def create_faiss_store(store_name: str, dimension: int, base_dir: str):
    """
    Creates or loads a FAISS index.
    """
    path = Path(base_dir) / store_name
    embedder = MedCPTDualEmbedder()
    
    if path.exists() and (path / "index.faiss").exists():
        print(f"[storage_utils] Loading existing FAISS index from: {path}")
        # Load existing
        return FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)
    
    print(f"[storage_utils] Index NOT found at {path}. Creating new empty index.")
    # Create new empty index
    index = faiss.IndexFlatIP(dimension)
    return FAISS(
        embedding_function=embedder,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
