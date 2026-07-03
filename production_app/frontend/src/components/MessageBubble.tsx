import type { ConversationMessage } from '../types'

interface Props {
  message: ConversationMessage
  onSelectEvidence: (message: ConversationMessage) => void
  isSelected: boolean
}

export default function MessageBubble({ message, onSelectEvidence, isSelected }: Props) {
  const { role, content, response, pending, error } = message

  if (role === 'user') {
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">{content}</div>
      </div>
    )
  }

  if (pending) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-pending">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-error">{error}</div>
      </div>
    )
  }

  if (!response) return null

  if (response.status !== 'answer') {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant bubble-abstain">
          <strong>No confident answer.</strong>
          <p>{describeAbstain(response.abstain_reason)}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="message message-assistant">
      <div className={`bubble bubble-assistant ${isSelected ? 'bubble-selected' : ''}`}>
        <div className="bubble-section">
          <h4>Clinician summary</h4>
          <ul>
            {response.clinician_bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        </div>
        {response.patient_bullets.length > 0 && (
          <div className="bubble-section">
            <h4>Patient summary</h4>
            <ul>
              {response.patient_bullets.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        )}
        <button className="evidence-link" onClick={() => onSelectEvidence(message)}>
          View {response.evidence_chunks.length} source{response.evidence_chunks.length === 1 ? '' : 's'} ·
          confidence {(response.confidence * 100).toFixed(0)}%
        </button>
      </div>
    </div>
  )
}

function describeAbstain(reason: string): string {
  const map: Record<string, string> = {
    out_of_scope: 'This question is outside ScholarBOT\'s knowledge domain (TB and pneumonia).',
    no_chunks_retrieved: 'No relevant evidence was found in the knowledge base.',
    evidence_insufficient_for_query: 'The retrieved evidence does not contain enough information to answer this question.',
    llm_abstain: 'The generation model could not produce a grounded answer from the evidence.',
    critique_rejected_all: 'All generated claims were rejected as unsupported by the evidence.',
    no_drug_chunks_retrieved: 'No relevant drug label evidence was found.',
    no_target_kb_for_summarize: 'Please upload a document before requesting a summary.',
    empty_store_for_summarize: 'The document appears to be empty or could not be read.',
  }
  if (map[reason]) return map[reason]
  if (reason.startsWith('low_confidence')) return 'Retrieved evidence did not meet the confidence threshold.'
  return 'Insufficient evidence to answer with confidence.'
}
