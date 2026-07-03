import { useCallback, useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import CitationsPanel from './components/CitationsPanel'
import { getHealth, resetSession, sendChatMessage, ApiError } from './api/client'
import type { ChatHistoryMessage, ConversationMessage, HealthResponse } from './types'

function newSessionId(): string {
  return `session-${Math.random().toString(36).slice(2)}-${Date.now()}`
}

function newMessageId(): string {
  return `msg-${Math.random().toString(36).slice(2)}`
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId)
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [sending, setSending] = useState(false)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [forceUserKb, setForceUserKb] = useState(false)
  const [hasUploadedDoc, setHasUploadedDoc] = useState(false)

  const refreshHealth = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h)
        setHealthError(null)
        setHasUploadedDoc(h.user_chunks > 0)
      })
      .catch((err) => setHealthError(err instanceof Error ? err.message : 'Backend unreachable.'))
  }, [])

  useEffect(() => {
    refreshHealth()
  }, [refreshHealth])

  const selectedMessage = messages.find((m) => m.id === selectedMessageId) ?? null

  async function handleSend(text: string) {
    const userMessage: ConversationMessage = { id: newMessageId(), role: 'user', content: text }
    const pendingId = newMessageId()
    const pendingMessage: ConversationMessage = { id: pendingId, role: 'assistant', content: '', pending: true }

    const history: ChatHistoryMessage[] = messages
      .filter((m) => !m.pending && !m.error)
      .map((m) => ({
        role: m.role,
        content: m.role === 'user' ? m.content : m.response?.clinician_bullets.join(' ') ?? '',
      }))

    setMessages((prev) => [...prev, userMessage, pendingMessage])
    setSending(true)

    try {
      const response = await sendChatMessage(text, sessionId, history, forceUserKb)
      setMessages((prev) => prev.map((m) => (m.id === pendingId ? { ...m, pending: false, response } : m)))
      setSelectedMessageId(pendingId)
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to reach ScholarBOT — please try again.'
      setMessages((prev) => prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m)))
    } finally {
      setSending(false)
    }
  }

  async function handleReset() {
    try {
      await resetSession(sessionId)
    } catch {
      // Non-fatal — starting a fresh session id is enough even if the backend call fails.
    }
    setSessionId(newSessionId())
    setMessages([])
    setSelectedMessageId(null)
  }

  function handleUploadComplete() {
    setHasUploadedDoc(true)
    setForceUserKb(true)
    refreshHealth()
  }

  return (
    <div className="app-shell">
      <Sidebar
        health={health}
        healthError={healthError}
        onReset={handleReset}
        onUploadComplete={handleUploadComplete}
        forceUserKb={forceUserKb}
        onToggleUserKb={setForceUserKb}
        hasUploadedDoc={hasUploadedDoc}
      />
      <ChatWindow
        messages={messages}
        onSend={handleSend}
        sending={sending}
        selectedMessageId={selectedMessageId}
        onSelectEvidence={(m) => setSelectedMessageId(m.id)}
      />
      <CitationsPanel message={selectedMessage} />
    </div>
  )
}
