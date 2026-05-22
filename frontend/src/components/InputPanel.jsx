import { useState } from 'react'
import { checkContradictions, checkLlmOnly } from '../api/client'

const METHODS = [
  { id: 'ensemble', label: 'Ensemble', hint: 'NLI pre-filter + LLM judge' },
  { id: 'llm',      label: 'LLM Only', hint: 'GPT judges the full document' },
]

/**
 * Two-textarea form that POSTs to /check or /check/llm-only depending on the
 * selected method, and surfaces the result via callbacks.
 *
 * Props:
 *   onReport(report|null) - called with the parsed JSON response (or null on reset)
 *   onLoading(bool)       - called when a request starts / finishes
 *   loading               - current in-flight state; disables submit while true
 */
export default function InputPanel({ onReport, onLoading, loading }) {
  const [context, setContext]   = useState('')
  const [response, setResponse] = useState('')
  const [method, setMethod]     = useState('ensemble')

  async function handleSubmit(e) {
    e.preventDefault()
    if (!context.trim() || !response.trim()) return

    onLoading(true)
    onReport(null)

    try {
      const fn = method === 'llm' ? checkLlmOnly : checkContradictions
      onReport(await fn(context, response))
    } catch (err) {
      onReport({ error: err.message })
    } finally {
      onLoading(false)
    }
  }

  const canSubmit = context.trim().length > 0 && response.trim().length > 0 && !loading
  const activeHint = METHODS.find(m => m.id === method).hint

  return (
    <form className="input-panel" onSubmit={handleSubmit}>
      <div className="input-group">
        <label htmlFor="context-input">Policy Document</label>
        <textarea
          id="context-input"
          value={context}
          onChange={e => setContext(e.target.value)}
          placeholder={`Paste your API docs, access policy, or technical specification here.\n\nExample:\n## Access Policy\nThe agent must never call DELETE or DROP endpoints.\nAccess to /admin/** is restricted to internal services only.\nAll database queries must be read-only. Write operations require human approval.`}
          rows={10}
        />
      </div>
      <div className="input-group">
        <label htmlFor="response-input">Agent's Planned Action</label>
        <textarea
          id="response-input"
          value={response}
          onChange={e => setResponse(e.target.value)}
          placeholder={`Paste the agent's reasoning or planned tool call here.\n\nExample:\nTo remove outdated records I will execute:\nDELETE /api/v1/records?status=expired\n\nThis will clean up 4,821 rows from the database.`}
          rows={10}
        />
      </div>

      <div className="method-row">
        <div className="method-toggle">
          {METHODS.map(m => (
            <button
              key={m.id}
              type="button"
              className={`method-toggle__btn${method === m.id ? ' method-toggle__btn--active' : ''}`}
              onClick={() => setMethod(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <span className="method-hint">{activeHint}</span>
      </div>

      <button type="submit" className="run-btn" disabled={!canSubmit}>
        Check for Violations
      </button>
    </form>
  )
}
