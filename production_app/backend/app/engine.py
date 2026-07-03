"""
Engine layer — the single object the API instantiates.
Orchestrates: context resolve → route → pipeline.run → structured response.
One instance is shared process-wide; ContextManager state is per-session,
keyed by session_id, so concurrent chat sessions don't bleed into each other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app import config, router as router_mod
from app.context import ContextManager
from app.pipeline import RAGPipeline


class ScholarBotEngine:
    def __init__(self, embedder=None, store_factory=None, pipeline: Optional[RAGPipeline] = None):
        """
        Args:
            embedder     : BGEEmbedder-like instance. Defaults to the module singleton.
            store_factory: callable(collection_name) -> ChromaStore-like. Defaults to
                           app.vectorstore.get_store. Overridable for tests.
            pipeline     : Pre-built RAGPipeline (for tests). If omitted, one is built
                           from a Retriever over the three named collections.
        """
        if embedder is None:
            from app.embedder import get_embedder
            embedder = get_embedder()
        self.embedder = embedder

        if store_factory is None:
            from app.vectorstore import get_store
            store_factory = get_store

        self.guidelines_store = store_factory(config.COLLECTION_GUIDELINES, embedder=self.embedder)
        self.druglabels_store = store_factory(config.COLLECTION_DRUGLABELS, embedder=self.embedder)
        self.user_store = store_factory(config.COLLECTION_USER, embedder=self.embedder)

        if pipeline is None:
            from app.retriever import Retriever
            retriever = Retriever(
                stores={
                    config.COLLECTION_GUIDELINES: self.guidelines_store,
                    config.COLLECTION_DRUGLABELS: self.druglabels_store,
                    config.COLLECTION_USER: self.user_store,
                },
                embedder=self.embedder,
            )
            pipeline = RAGPipeline(retriever)
        self.pipeline = pipeline

        self._sessions: Dict[str, ContextManager] = {}

    def _get_session(self, session_id: str) -> ContextManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = ContextManager()
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def reload_user_kb(self) -> None:
        """Call after a new PDF is ingested so the retriever sees the fresh index."""
        from app.vectorstore import get_store
        self.user_store = get_store(config.COLLECTION_USER, embedder=self.embedder)
        self.pipeline.retriever.reload_store(config.COLLECTION_USER, self.user_store)

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

        has_user_doc = self.user_store.count() > 0
        resolved = context_manager.resolve(query, history)

        route = router_mod.route(resolved, has_user_doc=has_user_doc, force_user_kb=force_user_kb)
        result = self.pipeline.run(resolved, route)
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
