# ScholarBOT frontend

React + TypeScript (Vite) chat UI for the ScholarBOT clinical RAG API — see
[production_app/README.md](../README.md) for the full project writeup.

## Development

```bash
npm install
npm run dev      # dev server on :5173, proxies /api -> http://localhost:8000
```

## Build

```bash
npm run build     # type-checks then builds to dist/
npm run lint
npm run preview   # serve the production build locally
```

Set `VITE_API_BASE_URL` (build-time env var) to point at a non-default backend URL.
