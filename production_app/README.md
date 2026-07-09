# ScholarBOT

**Try it live: [huggingface.co/spaces/AnushreeU/scholarbot](https://huggingface.co/spaces/AnushreeU/scholarbot)**

## What this is

At some point, everyone ends up Googling their symptoms. A clinician does
something similar, just with better sources — cross-referencing guidelines,
drug labels, and their own uploaded case notes instead of a search engine
built for everything else on the internet.

ScholarBOT is built for both of those people. Instead of an open-ended web
search, it queries a **curated knowledge base** in plain language — the kind
of question you'd actually type, not a boolean search string. Every answer
is grounded in that knowledge base: the specific passages it drew from are
shown alongside the answer as citations, split into a **clinician summary**
(technical, sourced) and a **patient summary** (plain-language, same
sources). If the retrieved evidence doesn't clear a confidence bar, the tool
**abstains** rather than guess — a wrong "I don't know" is annoying; a wrong
medical answer stated confidently is a different kind of problem.

The tool itself is a fairly simple RAG (retrieval-augmented generation)
system:
- **Retriever** — a dense embedding search (BGE-large-en-v1.5, 1024-dim) over
  the knowledge base, narrowed by a lightweight intent router, then
  re-ordered by a cross-encoder reranker (`ms-marco-MiniLM-L6-v2`) for
  precision. An earlier IR evaluation on this exact corpus found that adding
  classic keyword search (BM25) into the mix *hurt* ranking quality here, so
  retrieval is dense-only by design, not by omission.
- **Generator** — GPT-4o-mini, instructed to answer *only* from the retrieved
  passages and to output a structured abstain signal when it can't. A second
  self-critique pass reviews the draft against the evidence and strikes any
  claim that isn't actually supported before it reaches the user.
- **Deployed as** — a React/TypeScript chat UI talking to a FastAPI backend,
  Chroma as the vector store, containerized with Docker, with Kubernetes
  manifests for a local cluster and a live one-container deployment on
  Hugging Face Spaces.

As a proof of concept, the knowledge base currently covers two pulmonary
diseases — **tuberculosis** and **pneumonia** — plus their associated
medications. The scope is intentionally narrow: it's easier to demonstrate a
system abstains correctly *outside* its domain when that domain is well
defined. Source material comes from public clinical guidelines: **CDC**,
**WHO**, **ATS/IDSA**, **NICE (UK)**, the **British Thoracic Society**, and
**Medscape**, plus structured drug label data from **DailyMed** (FDA). See
[Knowledge base](#knowledge-base) below for exact figures.

Users can also upload their own PDF (a case report, a discharge summary,
whatever) and choose whether ScholarBOT answers from the curated knowledge
base, from their uploaded document only, or lean on both — each user's
upload is private to their own session; nobody else sees it.

## Core design principles

- **Fail-closed** — abstains when confidence is low rather than generating an uncertain answer
- **Evidence-only** — no outside knowledge is ever added; every sentence traces back to a retrieved passage
- **Dual-audience output** — a technical clinician summary and a plain-language patient summary for the same evidence
- **Full traceability** — every claim links to a knowledge-base document and page
- **Context retention** — follow-up questions with pronouns ("how is it treated?") resolve against conversation history before retrieval

## Repo layout

```
production_app/
├── backend/     FastAPI service — chunking, retrieval, routing, generation
├── frontend/    React + TypeScript chat UI (Vite)
├── docker/      Dockerfiles + docker-compose for local/demo deployment
└── k8s/         Kubernetes manifests (minikube / k3s)
```

This directory is a standalone, deployable rebuild of the ScholarBOT
pipeline from the parent repo (which runs as a Streamlit + FAISS app). The
two are independent — this rebuild swaps in a proper frontend/backend split,
Chroma instead of FAISS, tests, CI, and container/orchestration configs, but
doesn't touch or depend on the parent app.

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript, Vite |
| Backend | FastAPI (Python), Uvicorn |
| Vector store | Chroma (embedded, persistent) |
| Embedding model | `BAAI/bge-large-en-v1.5` (1024-dim, via `sentence-transformers`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| LLM | OpenAI `gpt-4o-mini` |
| Testing | pytest + pytest-cov, 80% coverage gate (currently ~95%) |
| CI | GitHub Actions (backend tests, frontend build/lint, Docker build smoke test) |
| Containerization | Docker (separate frontend/backend images for compose/k8s; a combined image for Spaces) |
| Orchestration | Kubernetes manifests (namespace, Deployments, Services, PVC, HPA, Ingress) |
| Live deployment | Hugging Face Spaces (Docker SDK) |

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

Request flow: `ContextManager` resolves pronouns/meta-references in the
query → `router` classifies domain + intent (a fast keyword path for clear
cases, LLM fallback for ambiguous ones) → `Retriever` does dense search
against the target collection(s) + cross-encoder rerank → `RAGPipeline`
generates an evidence-only answer and self-critiques it, abstaining rather
than guessing when evidence is thin. See `backend/app/` — each module is a
direct, tested port of the corresponding numbered file in the parent repo
(`01_config.py` → `config.py`, etc.).

## Knowledge base

Two Chroma collections, built from public guideline/drug-label exports:

| Collection | Chunks (source data) | Sources |
|---|---|---|
| `guidelines_kb` | 3,219 | CDC, WHO, ATS/IDSA, NICE (UK), British Thoracic Society, Medscape — TB and community-acquired pneumonia (CAP) clinical guidelines |
| `druglabels_kb` | 15,185 | DailyMed (FDA structured product labels) — TB and CAP medications |

Plus a per-session `user_kb_<hash>` collection for each uploaded document
(see [Multi-user document isolation](#multi-user-document-isolation)).

Rebuild the KB from source:

```bash
cd backend
python scripts/migrate_to_chroma.py \
  --guidelines ../../dataset/guidelines_chunks_cleaned.jsonl \
  --druglabels ../../dataset/druglabels_chunks.jsonl
```

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
class of race condition documented in [How I'd scale this](#how-id-scale-this)
for the engine singleton.

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
[How I'd scale this](#how-id-scale-this) for exactly when that tradeoff flips).

## Try the live demo

**https://huggingface.co/spaces/AnushreeU/scholarbot**

1. Ask a question directly — e.g. *"What are the symptoms of pneumonia?"* or
   *"What is the standard dosage of isoniazid for adults?"* — answered from
   the curated knowledge base, with citations in the sidebar.
2. Or upload a PDF, check **"Search my document only"**, and ask it to
   summarize the document or answer questions about it specifically.
3. Ask something outside TB/pneumonia (or with genuinely insufficient
   evidence) and watch it abstain instead of guessing.

## Running it yourself

### Backend

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt

# One-time: build the KB (see "Knowledge base" above)
python scripts/migrate_to_chroma.py \
  --guidelines ../../dataset/guidelines_chunks_cleaned.jsonl \
  --druglabels ../../dataset/druglabels_chunks.jsonl

export OPENAI_API_KEY=sk-...      # required for routing/generation; retrieval works without it
uvicorn app.main:app --reload --port 8000
```

Tests (mocked embedder/reranker/LLM — no GPU or network needed, runs in a few
seconds):

```bash
pytest    # enforces an 80% coverage gate (pytest.ini); currently ~95%
```

### Frontend

```bash
cd frontend
npm install
npm run dev     # :5173, proxies /api -> http://localhost:8000 (see vite.config.ts)
```

### Docker Compose (local full stack)

```bash
cd docker
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
# frontend: http://localhost:5173   backend: http://localhost:8000
```

### Kubernetes (minikube / k3s)

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

### Deploying your own copy to Hugging Face Spaces

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

## Known limitations

- **Cross-document comparison questions** ("what's the difference between X
  and Y?") are hard for this architecture — retrieval finds passages about X
  and passages about Y separately, but the evidence-only generation gate
  correctly refuses to synthesize a comparison unless a single retrieved
  passage actually draws one. It abstains rather than guesses, which is the
  intended fail-closed behavior, but it does mean genuinely comparative
  questions often come back empty.
- **Confidence score is an internal ranking signal, not a calibrated
  probability** — it's the raw cross-encoder reranker score, which is
  unbounded (not a 0–1 value). The UI clamps it for display; the underlying
  number is still just "did this pass the threshold," not "how sure is the
  model," and shouldn't be over-interpreted.
- **No cross-KB "search both curated data and my document" mode** —
  currently it's one or the other per query (see the `force_user_kb` toggle).
- See [Multi-user document isolation](#multi-user-document-isolation) for the
  orphaned-session-collection caveat.
