import type { ConversationMessage } from '../types'
import { formatConfidencePercent } from '../utils/format'

interface Props {
  message: ConversationMessage | null
}

export default function CitationsPanel({ message }: Props) {
  const response = message?.response

  return (
    <aside className="citations-panel">
      <h3>Sources</h3>
      {!response || response.status !== 'answer' ? (
        <p className="citations-empty">Select an answer to see its supporting evidence.</p>
      ) : (
        <>
          <div className="citations-meta">
            <span>Confidence: {formatConfidencePercent(response.confidence)}</span>
            <span>KB: {response.source || 'n/a'}</span>
          </div>
          <ol className="citations-list">
            {response.evidence_chunks.map((chunk) => (
              <li key={chunk.chunk_id} className="citation-item">
                <div className="citation-header">
                  [{chunk.chunk_id}] {chunk.citation}
                </div>
                <p className="citation-text">{chunk.text}</p>
              </li>
            ))}
          </ol>
        </>
      )}
    </aside>
  )
}
