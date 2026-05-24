import { useState, useRef } from 'react'
import { submitFeedback } from '../api/client'

function scoreColor(score) {
  if (score < 0.5) return 'var(--danger)'
  if (score < 0.75) return 'var(--warning)'
  return 'var(--success)'
}

const SEVERITY_COLORS = {
  blocking: 'var(--danger)',
  warning:  'var(--warning)',
  inferred: '#8b5cf6',
}

const METHOD_COLORS = {
  nli: '#3b82f6',
  llm: '#10b981',
  ensemble: '#8b5cf6',
}

function Badge({ label, color }) {
  return (
    <span className="badge" style={{ background: color }}>
      {label}
    </span>
  )
}

function FeedbackButtons({ runId, index }) {
  const [verdict, setVerdict] = useState(null)
  const submitted = useRef(false)

  async function handleFeedback(v) {
    if (!runId || submitted.current) return
    submitted.current = true
    setVerdict(v)
    try {
      await submitFeedback(runId, index, v)
    } catch {
      submitted.current = false
      setVerdict(null)
    }
  }

  if (!runId) return null

  return (
    <div className="feedback-buttons">
      <button
        className={`feedback-btn${verdict === 'confirmed' ? ' feedback-btn--active' : ''}`}
        onClick={() => handleFeedback('confirmed')}
        disabled={!!verdict}
        title="Confirm - this is a real violation"
      >
        ✓
      </button>
      <button
        className={`feedback-btn feedback-btn--fp${verdict === 'false_positive' ? ' feedback-btn--active' : ''}`}
        onClick={() => handleFeedback('false_positive')}
        disabled={!!verdict}
        title="False positive - not actually a violation"
      >
        ✗
      </button>
    </div>
  )
}

function ViolationItem({ v, index, runId }) {
  return (
    <li className="violation-item">
      <div className="violation-badges">
        <Badge label={v.severity} color={SEVERITY_COLORS[v.severity] ?? '#666'} />
        <Badge label={v.method.toUpperCase()} color={METHOD_COLORS[v.method] ?? '#666'} />
        <span className="confidence">{Math.round(v.confidence * 100)}% confident</span>
        <FeedbackButtons runId={runId} index={index} />
      </div>
      <div className="violation-spans">
        <div className="span-block span-block--response">
          <span className="span-label">Agent plans</span>
          <span className="span-text">"{v.response_span}"</span>
        </div>
        <div className="span-block span-block--context">
          <span className="span-label">Policy states</span>
          <span className="span-text">"{v.context_span}"</span>
        </div>
      </div>
      <p className="violation-explanation">{v.explanation}</p>
    </li>
  )
}

function AnalysisPanel({ report }) {
  if (!report.overall_reasoning) return null

  const showTokens = report.input_tokens > 0
  const violationsConfirmed = report.violations.length

  return (
    <div className="analysis-section">
      <div className="pipeline-row">
        <span className="pipeline-stat">
          <span className="pipeline-label">NLI</span>
          {report.nli_candidates} candidates
          {report.llm_escalations > 0 && (
            <> · {report.llm_escalations} uncertain</>
          )}
        </span>
        <span className="pipeline-arrow">→</span>
        <span className="pipeline-stat">
          {report.nli_candidates + report.llm_escalations} passed to LLM
        </span>
        <span className="pipeline-arrow">→</span>
        <span className="pipeline-stat">
          LLM confirmed {violationsConfirmed}
        </span>
      </div>

      {showTokens && (
        <div className="token-row">
          <span className="pipeline-label">TOKENS</span>
          {report.input_tokens.toLocaleString()} in · {report.output_tokens.toLocaleString()} out
          {report.cost_usd > 0 && (
            <span className="cost-value">
              &nbsp;&nbsp;COST&nbsp; ${report.cost_usd.toFixed(8)}
            </span>
          )}
        </div>
      )}

      <details className="reasoning-details">
        <summary>LLM Reasoning</summary>
        <p className="reasoning-block">{report.overall_reasoning}</p>
      </details>
    </div>
  )
}

/**
 * Displays the ViolationReport returned by POST /check.
 *
 * Props:
 *   report  - API response object, or null (empty state), or { error } (error state)
 *   loading - bool; shows spinner while request is in flight
 *   runId   - string; MongoDB run ID used to submit feedback (may be undefined)
 */
export default function ResultPanel({ report, loading, runId }) {
  if (loading) {
    return (
      <div className="result-panel result-panel--empty">
        <div className="spinner" />
        <p style={{ marginTop: '0.75rem' }}>Checking policy compliance…</p>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="result-panel result-panel--empty">
        <p>Run a check to see results.</p>
      </div>
    )
  }

  if (report.error) {
    return (
      <div className="result-panel result-panel--empty">
        <p className="error">{report.error}</p>
      </div>
    )
  }

  const color = scoreColor(report.compliance_score)

  return (
    <div className="result-panel">
      <div className="score-block">
        <div className="score-value" style={{ color }}>
          {Math.round(report.compliance_score * 100)}%
        </div>
        <div className="score-label">Compliance Score</div>
        <div className="score-meta">
          {report.method_used.toUpperCase()}
          {' · '}
          {report.nli_pairs_checked} pairs checked
          {' · '}
          {Math.round(report.processing_time_ms)}ms
        </div>
      </div>

      {report.violations.length === 0 ? (
        <p className="no-violations">No policy violations detected.</p>
      ) : (
        <ul className="violation-list">
          {report.violations.map((v, i) => (
            <ViolationItem
              key={v.response_span + v.context_span}
              v={v}
              index={i}
              runId={runId}
            />
          ))}
        </ul>
      )}

      <AnalysisPanel report={report} />
    </div>
  )
}
