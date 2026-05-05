import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/**
 * Two-textarea form that POSTs to /check and surfaces the result via callbacks.
 *
 * Props:
 *   onReport(report|null) — called with the parsed JSON response (or null on reset)
 *   onLoading(bool)       — called when a request starts / finishes
 */
export default function InputPanel({ onReport, onLoading }) {
  const [context, setContext] = useState('')
  const [response, setResponse] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    if (!context.trim() || !response.trim()) return

    onLoading(true)
    onReport(null)

    try {
      const res = await fetch(`${API_BASE}/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, response }),
      })
      if (!res.ok) throw new Error(`API returned ${res.status}`)
      onReport(await res.json())
    } catch (err) {
      onReport({ error: err.message })
    } finally {
      onLoading(false)
    }
  }

  const canSubmit = context.trim().length > 0 && response.trim().length > 0

  return (
    <form className="input-panel" onSubmit={handleSubmit}>
      <div className="input-group">
        <label htmlFor="context-input">Source Context</label>
        <textarea
          id="context-input"
          value={context}
          onChange={e => setContext(e.target.value)}
          placeholder="Paste the source document here…"
          rows={10}
        />
      </div>
      <div className="input-group">
        <label htmlFor="response-input">LLM Response</label>
        <textarea
          id="response-input"
          value={response}
          onChange={e => setResponse(e.target.value)}
          placeholder="Paste the LLM-generated response here…"
          rows={10}
        />
      </div>
      <button type="submit" className="run-btn" disabled={!canSubmit}>
        Detect Contradictions
      </button>
    </form>
  )
}
