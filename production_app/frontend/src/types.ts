export type ChatRole = 'user' | 'assistant'

export interface ChatHistoryMessage {
  role: ChatRole
  content: string
}

export interface EvidenceChunk {
  chunk_id: number
  text: string
  citation: string
  store: string
}

export interface ChatResponse {
  status: 'answer' | 'abstain' | 'error'
  abstain_reason: string
  clinician_bullets: string[]
  patient_bullets: string[]
  citations: string[]
  confidence: number
  evidence_chunks: EvidenceChunk[]
  source: string
}

export interface ConversationMessage {
  id: string
  role: ChatRole
  content: string
  response?: ChatResponse
  pending?: boolean
  error?: string
}

export interface HealthResponse {
  status: string
  guidelines_chunks: number
  druglabels_chunks: number
  user_chunks: number
}

export interface UploadResponse {
  added_chunks: number
  total_chars: number
  num_pages: number
  doc_name: string
}
