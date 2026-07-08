"""
Engine layer — the single object the API instantiates.
Orchestrates: context resolve → route → pipeline.run → structured response.
One instance is shared process-wide. Both ContextManager state and uploaded-
document storage are per-session, keyed by session_id, so concurrent users
never see each other's conversation history or uploaded documents.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple

from app import config, router as router_mod
from app.context import ContextManager
from app.pipeline import RAGPipeline


def _session_collection_name(session_id: str) -> str:
    """
    Deterministic, Chroma-safe collection name for a session's uploaded
    document. Hashed rather than sanitized directly: session_id is client-
    supplied, and Chroma collection names have strict constraints (length,
    allowed characters, must start/end alphanumeric) that arbitrary input
    could violate.
    """
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return f"{config.COLLECTION_USER}_{digest}"


class ScholarBotEngine:
    def __init__(self, embedder=None, store_factory=None, pipeline: Optional[RAGPipeline] = None):
        """
        Args:
            embedder     : BGEEmbedder-like instance. Defaults to the module singleton.
            store_factory: callable(collection_name, embedder=...) -> ChromaStore-like.
                           Defaults to app.vectorstore.get_store. Overridable for tests.
            pipeline     : Pre-built RAGPipeline (for tests). If omitted, one is built
                           from a Retriever over the shared guideline/druglabel collections
                           (per-session user-document collections are resolved per-request).
        """
        if embedder is None:
            from app.embedder import get_embedder
            embedder = get_embedder()
        self.embedder = embedder

        if store_factory is None:
            from app.vectorstore import get_store
            store_factory = get_store
        self._store_factory = store_factory

        self.guidelines_store = store_factory(config.COLLECTION_GUIDELINES, embedder=self.embedder)
        self.druglabels_store = store_factory(config.COLLECTION_DRUGLABELS, embedder=self.embedder)

        if pipeline is None:
            from app.retriever import Retriever
            retriever = Retriever(
                stores={
                    config.COLLECTION_GUIDELINES: self.guidelines_store,
                    config.COLLECTION_DRUGLABELS: self.druglabels_store,
                },
                embedder=self.embedder,
            )
            pipeline = RAGPipeline(retriever)
        self.pipeline = pipeline

        self._sessions: Dict[str, ContextManager] = {}
        self._user_stores: Dict[str, Any] = {}
        self._user_stores_lock = threading.Lock()

    def _get_session(self, session_id: str) -> ContextManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = ContextManager()
        return self._sessions[session_id]

    def get_user_store(self, session_id: str):
        """Returns this session's own uploaded-document store, creating it on first use."""
        if session_id not in self._user_stores:
            with self._user_stores_lock:
                if session_id not in self._user_stores:
                    self._user_stores[session_id] = self._store_factory(
                        _session_collection_name(session_id), embedder=self.embedder
                    )
        return self._user_stores[session_id]

    def reset_session(self, session_id: str) -> None:
        """Clears conversation history and forgets this session's uploaded document."""
        self._sessions.pop(session_id, None)
        store = self._user_stores.pop(session_id, None)
        if store is not None:
            store.delete_all()

    def generate_response(
        self,
        query: str,
        session_id: str = "default",
        force_user_kb: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """
        Returns (structured_answer, confidence, meta_dict). structured_answer keeps
        clinician/patient bullets and citations separate so the frontend renders
        its own citations panel instead of parsing markdown.
        """
        history = history or []
        context_manager = self._get_session(session_id)
        user_store = self.get_user_store(session_id)

        has_user_doc = user_store.count() > 0
        resolved = context_manager.resolve(query, history)

        route = router_mod.route(resolved, has_user_doc=has_user_doc, force_user_kb=force_user_kb)
        session_stores = {config.COLLECTION_USER: user_store}
        result = self.pipeline.run(resolved, route, session_stores=session_stores)
        context_manager.update_topic(resolved, result.status)

        answer = {
            "status": result.status,
            "abstain_reason": result.abstain_reason,
            "clinician_bullets": result.clinician_bullets,
            "patient_bullets": result.patient_bullets,
            "citations": result.citations,
        }
        meta = {
            "status": result.status,
            "abstain_reason": result.abstain_reason,
            "source": " + ".join(route.get("target_kbs", [])),
            "route": route,
            "evidence_chunks": result.evidence_chunks,
        }
        return answer, result.confidence, meta
