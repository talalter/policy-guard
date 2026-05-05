/** Color thresholds defined in CLAUDE.md */
function scoreColor(score) {
  if (score < 0.5) return 'var(--danger)'
  if (score < 0.75) return 'var(--warning)'
  return 'var(--success)'
}

const SEVERITY_COLORS = {
  direct: 'var(--danger)',
  partial: 'var(--warning)',
  'multi-hop': '#8b5cf6',
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

function ContradictionItem({ c }) {
  return (
    <li className="contradiction-item">
      <div className="contradiction-badges">
        <Badge label={c.severity} color={SEVERITY_COLORS[c.severity] ?? '#666'} />
        <Badge label={c.method.toUpperCase()} color={METHOD_COLORS[c.method] ?? '#666'} />
        <span className="confidence">{Math.round(c.confidence * 100)}% confident</span>
      </div>
      <div className="contradiction-spans">
        <div className="span-block span-block--response">
          <span className="span-label">Response says</span>
          <span className="span-text">"{c.response_span}"</span>
        </div>
        <div className="span-block span-block--context">
          <span className="span-label">Context says</span>
          <span className="span-text">"{c.context_span}"</span>
        </div>
      </div>
      <p className="contradiction-explanation">{c.explanation}</p>
    </li>
  )
}

/**
 * Displays the ContradictionReport returned by POST /check.
 *
 * Props:
 *   report  — API response object, or null (empty state), or { error } (error state)
 *   loading — bool; shows spinner while request is in flight
 */
export default function ResultPanel({ report, loading }) {
  if (loading) {
    return (
      <div className="result-panel result-panel--empty">
        <div className="spinner" />
        <p style={{ marginTop: '0.75rem' }}>Analyzing…</p>
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

  const color = scoreColor(report.faithfulness_score)

  return (
    <div className="result-panel">
      <div className="score-block">
        <div className="score-value" style={{ color }}>
          {Math.round(report.faithfulness_score * 100)}%
        </div>
        <div className="score-label">Faithfulness Score</div>
        <div className="score-meta">
          {report.method_used.toUpperCase()}
          {' · '}
          {report.nli_pairs_checked} pairs checked
          {' · '}
          {Math.round(report.processing_time_ms)}ms
        </div>
      </div>

      {report.contradictions.length === 0 ? (
        <p className="no-contradictions">No contradictions detected.</p>
      ) : (
        <ul className="contradiction-list">
          {report.contradictions.map((c) => (
            <ContradictionItem key={c.response_span + c.context_span} c={c} />
          ))}
        </ul>
      )}
    </div>
  )
}
