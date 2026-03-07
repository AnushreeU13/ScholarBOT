"""
Deduplication utilities for RAG system.
Uses SHA-256 hashing to detect duplicate documents.
"""

import hashlib
import json
import os
from typing import Optional, Dict
from pathlib import Path


class DocumentTracker:
    """Track processed documents to avoid duplicates."""
    
    def __init__(self, metadata_file: str = "metadata/document_tracker.json"):
        """
        Initialize document tracker.
        
        Args:
            metadata_file: Path to JSON file storing document hashes
        """
        self.metadata_file = metadata_file
        self.tracked_docs = {}
        
        # Load existing tracking data
        self._load()
    
    def _load(self):
        """Load existing document tracking data."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.tracked_docs = json.load(f)
    
    def _compute_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of document content.
        
        Args:
            content: Document text content
        
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def is_processed(self, content: str, source_path: str = None):
        """
        Check if document has been processed.
        
        Args:
            content: Document text content
            source_path: Optional source file path
        
        Returns:
            Tuple of (is_processed, existing_metadata or None)
        """
        doc_hash = self._compute_hash(content)
        
        if doc_hash in self.tracked_docs:
            return True, self.tracked_docs[doc_hash]
        
        return False, None
    
    def add_document(self, content: str, source_path: str, document_name: str, 
                    chunk_count: int, kb_name: str, page_count: int = None):
        """
        Add document to tracking system.
        
        Args:
            content: Document text content
            source_path: Source file path
            document_name: Name of the document
            chunk_count: Number of chunks created
            kb_name: Which KB it was added to ('primary_KB' or 'user_fact_KB')
            page_count: Number of pages (if applicable)
        """
        doc_hash = self._compute_hash(content)
        
        self.tracked_docs[doc_hash] = {
            'hash': doc_hash,
            'source_path': source_path,
            'document_name': document_name,
            'chunk_count': chunk_count,
            'kb_name': kb_name,
            'page_count': page_count,
            'processed_at': None  # Can add timestamp if needed
        }
    
    def update_document(self, content: str, source_path: str, document_name: str,
                       chunk_count: int, kb_name: str, page_count: int = None):
        """
        Update document entry (for re-indexing).
        
        Args:
            content: Document text content
            source_path: Source file path
            document_name: Name of the document
            chunk_count: Number of chunks created
            kb_name: Which KB it was added to
            page_count: Number of pages (if applicable)
        """
        doc_hash = self._compute_hash(content)
        
        if doc_hash in self.tracked_docs:
            # Update existing entry
            self.tracked_docs[doc_hash].update({
                'source_path': source_path,
                'document_name': document_name,
                'chunk_count': chunk_count,
                'kb_name': kb_name,
                'page_count': page_count
            })
        else:
            # Add new entry
            self.add_document(content, source_path, document_name, chunk_count, kb_name, page_count)
    
    def save(self):
        """Save tracking data to disk."""
        os.makedirs(os.path.dirname(self.metadata_file) if os.path.dirname(self.metadata_file) else '.', exist_ok=True)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.tracked_docs, f, indent=2, ensure_ascii=False)
    
    def get_stats(self) -> Dict:
        """Get statistics about tracked documents."""
        return {
            'total_documents': len(self.tracked_docs),
            'by_kb': {}
        }

