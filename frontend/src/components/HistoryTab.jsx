import { useState, useEffect, useCallback } from 'react'
import { fetchHistory, fetchStats, fetchHistoryItem, submitFeedback, deleteHistoryItem } from '../api/client'

function scoreColor(score) {
  if (score < 0.5) return 'var(--danger)'
  if (score < 0.75) return 'var(--warning)'
  return 'var(--success)'
}

function severityColor(severity) {
  if (severity === 'direct') return 'var(--danger)'
  if (severity === 'partial') return 'var(--warning)'
  return '#60a5fa'
}

function StatsBar({ stats }) {
  return (
    <div className="stats-bar">
      <span className="stats-item">
        <strong>{stats.total_runs}</strong> runs
      </span>
      <span className="stats-sep">·</span>
      <span className="stats-item">
        <strong>{stats.total_contradictions}</strong> contradictions
      </span>
      <span className="stats-sep">·</span>
      <span className="stats-item">
        <strong>{Math.round(stats.confirmed_rate * 100)}%</strong> confirmed by feedback
      </span>
    </div>
  )
}

function ContradictionRow({ contradiction: c, index, runId }) {
  const [verdict, setVerdict] = useState(null)

  async function handleFeedback(v) {
    if (verdict) return
    setVerdict(v)
    try {
      await submitFeedback(runId, index, v)
    } catch {
      setVerdict(null)
    }
  }

  return (
    <li className="contradiction-item">
      <div className="contradiction-badges">
        <span className="badge" style={{ background: severityColor(c.severity) }}>
          {c.severity}
        </span>
        <span className="badge" style={{ background: 'var(--surface-2)', color: 'var(--text-muted)' }}>
          {c.method}
        </span>
        <span className="confidence">{Math.round(c.confidence * 100)}% confidence</span>
        <div className="feedback-buttons">
          <button
            className={`feedback-btn${verdict === 'confirmed' ? ' feedback-btn--active' : ''}`}
            onClick={() => handleFeedback('confirmed')}
            disabled={!!verdict}
            title="Confirm — this is a real contradiction"
          >✓</button>
          <button
            className={`feedback-btn feedback-btn--fp${verdict === 'false_positive' ? ' feedback-btn--active' : ''}`}
            onClick={() => handleFeedback('false_positive')}
            disabled={!!verdict}
            title="False positive — not actually a contradiction"
          >✗</button>
        </div>
      </div>
      <div className="contradiction-spans">
        <div className="span-block span-block--response">
          <span className="span-label">Agent said</span>
          <span className="span-text">{c.response_span}</span>
        </div>
        <div className="span-block span-block--context">
          <span className="span-label">Policy says</span>
          <span className="span-text">{c.context_span}</span>
        </div>
      </div>
      <p className="contradiction-explanation">{c.explanation}</p>
    </li>
  )
}

function RunDetailModal({ runId, onClose }) {
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchHistoryItem(runId)
      .then(setDetail)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [runId])

  const handleBackdropClick = useCallback(
    (e) => { if (e.target === e.currentTarget) onClose() },
    [onClose]
  )

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal" role="dialog" aria-modal="true">
        <div className="modal-header">
          <span className="modal-title">Run Detail</span>
          <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="modal-body">
          {loading && <div className="spinner" style={{ margin: '2rem auto' }} />}
          {error && <p className="error">{error}</p>}
          {detail && (
            <>
              <div className="modal-meta-row">
                <span style={{ color: scoreColor(detail.faithfulness_score), fontWeight: 700, fontSize: 22 }}>
                  {Math.round(detail.faithfulness_score * 100)}%
                </span>
                <span className="modal-badge">{detail.method_used.toUpperCase()}</span>
                <span className="modal-badge">{detail.provider}</span>
                <span className="modal-timestamp">
                  {new Date(detail.timestamp).toLocaleString()}
                </span>
              </div>

              <div>
                <p className="modal-section-label">Policy Document</p>
                <pre className="modal-text-block">{detail.context}</pre>
              </div>

              <div>
                <p className="modal-section-label">Agent Action</p>
                <pre className="modal-text-block">{detail.response}</pre>
              </div>

              <div>
                <p className="modal-section-label">
                  Contradictions ({detail.contradictions.length})
                </p>
                {detail.contradictions.length === 0 ? (
                  <p className="modal-no-violations">No violations detected</p>
                ) : (
                  <ul className="contradiction-list">
                    {detail.contradictions.map((c, i) => (
                      <ContradictionRow
                        key={i}
                        contradiction={c}
                        index={i}
                        runId={detail.run_id}
                      />
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Displays the last 50 detection runs and aggregate stats.
 * Clicking any row opens a modal with the full context, response, and violations.
 */
export default function HistoryTab() {
  const [history, setHistory] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [pendingDelete, setPendingDelete] = useState(null)

  async function handleDelete(e, runId) {
    e.stopPropagation()
    if (pendingDelete !== runId) {
      setPendingDelete(runId)
      return
    }
    try {
      await deleteHistoryItem(runId)
      setHistory((prev) => prev.filter((item) => item.run_id !== runId))
      if (selectedRunId === runId) setSelectedRunId(null)
      fetchStats().then(setStats).catch(() => {})
    } catch {
      /* leave row in place if delete fails */
    } finally {
      setPendingDelete(null)
    }
  }

  useEffect(() => {
    async function load() {
      try {
        const [historyData, statsData] = await Promise.allSettled([
          fetchHistory(),
          fetchStats(),
        ])
        if (historyData.status === 'rejected') {
          throw new Error(`History unavailable — is MONGODB_URL set?`)
        }
        setHistory(historyData.value)
        if (statsData.status === 'fulfilled') setStats(statsData.value)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="history-tab history-tab--center">
        <div className="spinner" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="history-tab history-tab--center">
        <p className="error">{error}</p>
      </div>
    )
  }

  return (
    <div className="history-tab">
      {stats && <StatsBar stats={stats} />}

      {history.length === 0 ? (
        <p className="history-empty">No runs yet — run a check to see history here.</p>
      ) : (
        <>
          <p className="history-hint">Click any row to view full context and response.</p>
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Score</th>
                <th>Contradictions</th>
                <th>Method</th>
                <th>Provider</th>
                <th>Context</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr
                  key={item.run_id}
                  className="history-row-clickable"
                  onClick={() => setSelectedRunId(item.run_id)}
                >
                  <td className="metric-cell" style={{ whiteSpace: 'nowrap' }}>
                    {new Date(item.timestamp).toLocaleString()}
                  </td>
                  <td className="metric-cell" style={{ color: scoreColor(item.faithfulness_score) }}>
                    {Math.round(item.faithfulness_score * 100)}%
                  </td>
                  <td className="metric-cell">{item.contradiction_count}</td>
                  <td>{item.method_used.toUpperCase()}</td>
                  <td>{item.provider}</td>
                  <td className="context-snippet">{item.context_snippet}</td>
                  <td onClick={(e) => e.stopPropagation()} style={{ width: 60, textAlign: 'right' }}>
                    <button
                      className={`delete-btn${pendingDelete === item.run_id ? ' delete-btn--confirm' : ''}`}
                      onClick={(e) => handleDelete(e, item.run_id)}
                      onBlur={() => setPendingDelete(null)}
                      title={pendingDelete === item.run_id ? 'Click again to confirm' : 'Delete run'}
                    >
                      {pendingDelete === item.run_id ? 'Sure?' : '✕'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {selectedRunId && (
        <RunDetailModal
          runId={selectedRunId}
          onClose={() => setSelectedRunId(null)}
        />
      )}
    </div>
  )
}
