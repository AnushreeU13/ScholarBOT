/**
 * The backend's `confidence` field is the raw cross-encoder reranker score
 * for QA/drug-info answers (unbounded — a real logit-like value, not a 0-1
 * probability; frequently > 1) and a hardcoded 1.0 for summarize answers.
 * It's meaningful as an internal ranking/threshold signal, but showing it
 * directly as "619%" in the UI is misleading. Clamp to [0, 1] for display.
 */
export function formatConfidencePercent(confidence: number): string {
  const clamped = Math.min(Math.max(confidence, 0), 1)
  return `${(clamped * 100).toFixed(0)}%`
}
