"""
Semantic chunking utilities for RAG system.
Chunks text into 400 token chunks with 50 token overlap (defaults).
"""

import re
from typing import List, Dict
from transformers import AutoTokenizer

# Module-level tokenizer cache — avoids reloading SciBERT once per page
_tokenizer_cache: dict = {}

def _get_tokenizer(model_name: str):
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def semantic_chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    model_name: str = "allenai/scibert_scivocab_uncased"
) -> List[Dict[str, any]]:
    """
    Chunk text semantically with token-based sizing.
    
    Args:
        text: Input text to chunk
        chunk_size: Number of tokens per chunk (default: 700)
        overlap: Number of overlapping tokens between chunks (default: 100)
        model_name: Tokenizer model name for token counting
    
    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    if text is None:
        text = ""
    text = str(text) # Force string conversion
    
    tokenizer = _get_tokenizer(model_name)
    
    # Split text into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        # Count tokens in sentence
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        # If adding this sentence exceeds chunk size, save current chunk
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'token_count': current_tokens
            })
            
            # Start new chunk with overlap
            if overlap > 0:
        # Build overlap by taking as many trailing sentences as needed
                # to reach (approximately) the requested overlap token budget.
                tail_sentences = []
                tail_tokens = 0
                for sent in reversed(current_chunk):
                    t = len(tokenizer.encode(sent, add_special_tokens=False))
                    # If we already have some overlap, stop when adding would exceed budget
                    if tail_sentences and (tail_tokens + t) > overlap:
                        break
                    tail_sentences.insert(0, sent)
                    tail_tokens += t
                    if tail_tokens >= overlap:
                        break

                overlap_text = ' '.join(tail_sentences).strip()
                overlap_tokens = len(tokenizer.encode(overlap_text, add_special_tokens=False))

                # Fallback: ensure at least last sentence is preserved
                if not overlap_text:
                    overlap_text = current_chunk[-1]
                    overlap_tokens = len(tokenizer.encode(overlap_text, add_special_tokens=False))

                current_chunk = [overlap_text]
                current_tokens = overlap_tokens


            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'token_count': current_tokens
        })
    
    return chunks


def chunk_document(
    text: str,
    document_name: str,
    page_number: int = None,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, any]]:
    """
    Chunk a document and add metadata.
    
    Args:
        text: Document text
        document_name: Name of the document
        page_number: Page number (if applicable)
        chunk_size: Tokens per chunk
        overlap: Overlapping tokens
    
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = semantic_chunk_text(text, chunk_size, overlap)
    
    # Add metadata to each chunk
    for idx, chunk in enumerate(chunks):
        chunk['metadata'] = {
            'document_name': document_name,
            'chunk_index': idx,
            'page_number': page_number,
            'total_chunks': len(chunks)
        }
    
    return chunks
