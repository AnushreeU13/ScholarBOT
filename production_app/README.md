# ScholarBOT — production rebuild

A production-shaped rebuild of the ScholarBOT clinical RAG pipeline (evidence-only
Q&A over WHO/CDC TB and pneumonia guidelines + drug labels, built in the parent
repo). This directory is a standalone, deployable app: React/TypeScript frontend,
FastAPI backend, Chroma vector store, pytest CI, Docker, and Kubernetes manifests.
The original Streamlit + FAISS app one level up is unaffected — this is a parallel
rebuild for portfolio/skill purposes, not a replacement.

```
production_app/
├── backend/     FastAPI service — chunking, retrieval, routing, generation
├── frontend/    React + TypeScript chat UI (Vite)
├── docker/      Dockerfiles + docker-compose for local/demo deployment
└── k8s/         Kubernetes manifests (minikube / k3s)
```

## Architecture

```
 ┌─────────────┐      /chat, /upload,      ┌──────────────────┐
 │   React UI  │ ───► /health, /session ──► │   FastAPI backend │
 │ (chat + cite│ ◄─────────────────────────  │                   │
 │  panel)     │                             │ context → router  │
 └─────────────┘                             │ → retriever →      │
                                              │   pipeline (LLM)   │
                                              └─────────┬─────────┘
                                                         │
                                              ┌──────────▼─────────┐
                                              │   Chroma (vector    │
                                              │   store, per-KB     │
                                              │   collections)      │
                                              └─────────────────────┘
```

Request flow mirrors the original pipeline: `ContextManager` resolves pronouns/
meta-references → `router` classifies domain + intent (keyword fast path, LLM
fallback for ambiguous queries) → `Retriever` does dense search per target
collection + cross-encoder rerank → `RAGPipeline` generates an evidence-only
answer and self-critiques it, abstaining rather than guessing when evidence is
thin. See `backend/app/` — each module is a direct, tested port of the
corresponding numbered file in the parent repo (`01_config.py` → `config.py`,
etc.), with FAISS/BM25 swapped for Chroma dense-only retrieval (the parent
repo's own IR eval found BM25/RRF hurt ranking quality on this corpus, so
dense-only + rerank was kept as-is here).

## Multi-user document isolation

Uploaded documents are per-session, not global. Each `session_id` gets its own
Chroma collection (`user_kb_<sha256(session_id)[:32]>` — hashed rather than a
direct pass-through, since `session_id` is client-supplied and Chroma
collection names have strict character/length constraints). `ContextManager`
conversation state was already keyed by `session_id`; `ScholarBotEngine`
extends the same pattern to uploaded-document storage
(`get_user_store(session_id)`), so two people using the app at once never see
each other's document or conversation history. "Clear conversation" in the UI
calls `POST /session/{id}/reset`, which drops both.

Retrieval reads for the shared guideline/drug-label collections and a
session's own document collection happen through the same `Retriever`
instance without any per-request mutation of shared state (session stores are
passed in as a `session_stores` override on each call) — avoiding the same
class of race condition documented in "How I'd scale this" for the engine
singleton.

One open item: session collections aren't garbage-collected on their own — a
session that's abandoned without hitting "Clear conversation" leaves an empty
or small orphaned Chroma collection behind. Fine for a demo; a real deployment
would want a TTL-based sweep (e.g. a periodic job dropping collections idle
longer than N hours).

## Vector store: Chroma vs. Pinecone

Chroma was chosen for this rebuild. Documented tradeoff:

| | **Chroma** (chosen) | **Pinecone** |
|---|---|---|
| Hosting | Self-hosted / embedded — runs in-process or as a local server, no external account | Managed SaaS — signup, API key, network dependency |
| Cost | Free, no usage limits | Free tier is capped (pods/storage); scales to paid tiers |
| Local dev & CI | Trivial — `chromadb.EphemeralClient()` gives an in-memory store for tests, `PersistentClient` for a local volume | Requires network access even for local dev; CI needs a real (or mocked) Pinecone project |
| Portfolio/demo fit | Whole stack is reproducible from a fresh clone + `docker compose up`, no third-party account needed | Better demonstrates working with a managed cloud vector DB, which is common in real production stacks |
| Scaling | Single-writer persistent client by default; scaling reads/writes across nodes needs the separate Chroma server deployment or a different store | Built for horizontal scale, replicas, and multi-region out of the box — no operational work on our side |
| Metadata filtering | Supported, same shape as this app already uses (`document_name`, `page_number`, etc.) | Supported, broadly equivalent for this use case |

For a self-contained, free, easy-to-run demo — the goal here — Chroma wins.
For a real multi-tenant production deployment with unpredictable scale,
Pinecone's managed infrastructure removes an entire operational burden (see
"How I'd scale this" below for exactly when that tradeoff flips).

## Backend

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt

# One-time: migrate the existing guideline/drug-label KB into Chroma
python scripts/migrate_to_chroma.py \
  --guidelines ../../dataset/guidelines_chunks_cleaned.jsonl \
  --druglabels ../../dataset/druglabels_chunks.jsonl

export OPENAI_API_KEY=sk-...      # required for routing/generation; retrieval works without it
uvicorn app.main:app --reload --port 8000
```

Tests (mocked embedder/reranker/LLM — no GPU or network needed, runs in a few
seconds):

```bash
pytest    # enforces an 80% coverage gate (pytest.ini); currently ~94%
```

## Frontend

```bash
cd frontend
npm install
npm run dev     # :5173, proxies /api -> http://localhost:8000 (see vite.config.ts)
```

## Docker Compose (local full stack)

```bash
cd docker
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
# frontend: http://localhost:5173   backend: http://localhost:8000
```

## Kubernetes (minikube / k3s)

```bash
# minikube
minikube start
minikube addons enable ingress
minikube addons enable metrics-server
eval $(minikube docker-env)   # build images directly into minikube's Docker

docker build -f docker/Dockerfile.backend -t scholarbot-backend:local backend
docker build -f docker/Dockerfile.frontend -t scholarbot-frontend:local frontend

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/backend-configmap.yaml
kubectl create secret generic scholarbot-backend-secret \
  --namespace scholarbot --from-literal=OPENAI_API_KEY=sk-...
kubectl apply -f k8s/backend-pvc.yaml -f k8s/backend-deployment.yaml -f k8s/backend-service.yaml
kubectl apply -f k8s/frontend-deployment.yaml -f k8s/frontend-service.yaml -f k8s/frontend-hpa.yaml
kubectl apply -f k8s/ingress.yaml

echo "$(minikube ip) scholarbot.local" | sudo tee -a /etc/hosts
# -> http://scholarbot.local
```

k3s: same manifests apply directly (Traefik ships built in — change
`ingressClassName: nginx` to `traefik` in `k8s/ingress.yaml`, or drop the
Ingress and use k3s's `svclb` + a NodePort Service instead).

## Deploying to Hugging Face Spaces

The two-container layout (separate frontend/backend images) doesn't fit a
Space, which runs one container on one port. `docker/Dockerfile.spaces`
bundles the built frontend as static files served directly by FastAPI:

1. Create a new Space → SDK: **Docker**.
2. Push this repo (or just `production_app/`) to the Space's git remote, with
   `docker/Dockerfile.spaces` as the Space's Dockerfile (set `# Dockerfile`
   path in the Space's README frontmatter if it isn't at the repo root).
3. Add `OPENAI_API_KEY` as a Space secret.
4. On first boot, exec into the running Space (or add a one-time init step)
   and run `python scripts/migrate_to_chroma.py ...` to populate the
   `/data/chroma` volume — Spaces persist `/data` across restarts on paid
   hardware tiers; on the free tier, bake the Chroma DB into the image at
   build time instead (`COPY` a pre-built `chroma_data/` dir) since the
   filesystem doesn't persist between restarts there.

## How I'd scale this

**Backend is the bottleneck, not the frontend.** The frontend is stateless
and trivially scales horizontally (see `k8s/frontend-hpa.yaml`). The backend
currently runs as a single replica because:
- Chroma's `PersistentClient` is a single-process embedded store, backed by a
  `ReadWriteOnce` PVC — a second pod can't safely share it.
- `ScholarBotEngine` holds per-session `ContextManager` state *and* per-session
  uploaded-document stores in in-memory dicts, so a second replica wouldn't see
  a session (or its uploaded document) created on the first.

To scale past one backend replica, in order of effort:
1. **Move session state out of the process** — Redis-backed `ContextManager`
   storage keyed by `session_id`, so any replica can serve any request.
2. **Move Chroma to its own service** — run `chromadb`'s server mode (or
   switch to Pinecone/a managed store, per the tradeoff above) so multiple
   backend replicas share one vector store over the network instead of a
   local PVC.
3. **Split embedding/reranking onto their own service** — `sentence-transformers`
   model loads are the heaviest part of a pod's memory/startup time; running
   them as a separate internal service (or batching requests) lets the API
   layer scale independently of GPU/CPU-heavy inference, and enables a GPU
   node pool sized just for that service.
4. With (1) and (2) done, `scholarbot-backend` becomes a normal stateless
   Deployment — add an HPA on CPU/request-latency, same as the frontend.
5. **Cache aggressively at the router/retrieval layer** — the router's
   keyword fast path already avoids an LLM call for most queries; a
   request-level cache (query hash → route/retrieval result) would cut
   further cost on repeated or near-duplicate questions across users.
