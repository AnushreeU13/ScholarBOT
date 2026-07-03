import { useEffect, useRef } from 'react'
import type { ConversationMessage } from '../types'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'

interface Props {
  messages: ConversationMessage[]
  onSend: (text: string) => void
  sending: boolean
  selectedMessageId: string | null
  onSelectEvidence: (message: ConversationMessage) => void
}

export default function ChatWindow({ messages, onSend, sending, selectedMessageId, onSelectEvidence }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length])

  return (
    <main className="chat-window">
      <div className="chat-history">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask about TB or pneumonia diagnosis, treatment, prevention, or drug dosing.</p>
            <p className="empty-state-hint">Every answer is grounded in cited guideline or drug-label evidence.</p>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble
            key={m.id}
            message={m}
            onSelectEvidence={onSelectEvidence}
            isSelected={m.id === selectedMessageId}
          />
        ))}
        <div ref={bottomRef} />
      </div>
      <ChatInput onSend={onSend} disabled={sending} />
    </main>
  )
}
