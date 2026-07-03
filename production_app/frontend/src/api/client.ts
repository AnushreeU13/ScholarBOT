import type { ChatHistoryMessage, ChatResponse, HealthResponse, UploadResponse } from '../types'

// In dev, Vite proxies /api -> the backend (see vite.config.ts). In production,
// the frontend is served behind a reverse proxy / ingress that routes /api the
// same way — see production_app/k8s and the Docker Compose config.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })

  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail.detail ?? res.statusText)
  }

  return res.json() as Promise<T>
}

export function sendChatMessage(
  query: string,
  sessionId: string,
  history: ChatHistoryMessage[],
  forceUserKb: boolean,
): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({ query, session_id: sessionId, history, force_user_kb: forceUserKb }),
  })
}

export function resetSession(sessionId: string): Promise<{ status: string }> {
  return request(`/session/${encodeURIComponent(sessionId)}/reset`, { method: 'POST' })
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${BASE_URL}/upload`, { method: 'POST', body: formData })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail.detail ?? res.statusText)
  }
  return res.json() as Promise<UploadResponse>
}

export { ApiError }
