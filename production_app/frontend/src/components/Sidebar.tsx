import { useRef, useState } from 'react'
import type { HealthResponse } from '../types'
import { uploadDocument } from '../api/client'

interface Props {
  sessionId: string
  health: HealthResponse | null
  healthError: string | null
  onReset: () => void
  onUploadComplete: (docName: string, addedChunks: number) => void
  forceUserKb: boolean
  onToggleUserKb: (value: boolean) => void
  hasUploadedDoc: boolean
}

export default function Sidebar({
  sessionId,
  health,
  healthError,
  onReset,
  onUploadComplete,
  forceUserKb,
  onToggleUserKb,
  hasUploadedDoc,
}: Props) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setUploadError(null)
    try {
      const result = await uploadDocument(file, sessionId)
      onUploadComplete(result.doc_name, result.added_chunks)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <aside className="sidebar">
      <h1 className="brand">ScholarBOT</h1>
      <p className="brand-subtitle">Evidence-only clinical assistant · TB &amp; Pneumonia</p>

      <section className="sidebar-section">
        <h3>Upload document</h3>
        <input ref={fileInputRef} type="file" accept="application/pdf" onChange={handleFileChange} disabled={uploading} />
        {uploading && <p className="hint">Ingesting…</p>}
        {uploadError && <p className="error-text">{uploadError}</p>}
      </section>

      <section className="sidebar-section">
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={forceUserKb}
            disabled={!hasUploadedDoc}
            onChange={(e) => onToggleUserKb(e.target.checked)}
          />
          Search my document only
        </label>
      </section>

      <section className="sidebar-section">
        <h3>Knowledge base</h3>
        {healthError && <p className="error-text">{healthError}</p>}
        {health && (
          <ul className="kb-stats">
            <li>Guidelines: {health.guidelines_chunks.toLocaleString()} chunks</li>
            <li>Drug labels: {health.druglabels_chunks.toLocaleString()} chunks</li>
            <li>Your document: {health.user_chunks.toLocaleString()} chunks</li>
          </ul>
        )}
      </section>

      <button className="reset-button" onClick={onReset}>
        Clear conversation
      </button>

      <p className="disclaimer">
        Fail-closed design — ScholarBOT abstains rather than guesses when evidence is insufficient.
      </p>
    </aside>
  )
}
