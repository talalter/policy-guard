import { useState, useEffect } from 'react'
import { fetchBenchmarkDatasets, fetchBenchmarkResults } from '../api/client'

const METHOD_LABELS = {
  nli: 'NLI only',
  llm: 'GPT-5.4-mini only',
  ensemble: 'Ensemble',
}

const DIFFICULTY_ORDER = ['easy', 'medium', 'hard']
const DIFFICULTY_N = { easy: 30, medium: 55, hard: 35 }

function PctCell({ value }) {
  return <td className="metric-cell">{(value * 100).toFixed(1)}%</td>
}

function ResultRow({ r }) {
  return (
    <tr>
      <td className="method-cell">{METHOD_LABELS[r.method] ?? r.method}</td>
      <PctCell value={r.precision} />
      <PctCell value={r.recall} />
      <PctCell value={r.f1} />
      <td className="metric-cell">{(r.avg_latency_ms / 1000).toFixed(2)}s</td>
      <td className="metric-cell">${r.estimated_cost_per_call.toFixed(4)}</td>
    </tr>
  )
}

export default function BenchmarkTab() {
  const [datasets, setDatasets] = useState([])
  const [selectedKey, setSelectedKey] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load available datasets once on mount
  useEffect(() => {
    fetchBenchmarkDatasets()
      .then(data => {
        setDatasets(data)
        if (data.length > 0) setSelectedKey(data[0].key)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  // Fetch results whenever the selected dataset changes
  useEffect(() => {
    if (!selectedKey) return
    setLoading(true)
    setResults(null)
    fetchBenchmarkResults(selectedKey)
      .then(data => {
        setResults(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [selectedKey])

  if (error) {
    return (
      <div className="benchmark-tab">
        <p className="error">Could not load benchmark results: {error}</p>
        <p style={{ marginTop: '0.75rem', color: 'var(--text-muted)', fontSize: '13px' }}>
          Generate results first: <code>python -m backend.tools.benchmark</code>
        </p>
      </div>
    )
  }

  if (loading || !results) {
    return (
      <div className="benchmark-tab" style={{ display: 'flex', justifyContent: 'center', paddingTop: '4rem' }}>
        <div className="spinner" />
      </div>
    )
  }

  return (
    <div className="benchmark-tab">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
        <h2 style={{ margin: 0 }}>Detection Method Comparison</h2>
        {datasets.length > 1 && (
          <div className="dataset-toggle">
            {datasets.map(ds => (
              <button
                key={ds.key}
                className={`dataset-btn${selectedKey === ds.key ? ' active' : ''}`}
                onClick={() => setSelectedKey(ds.key)}
              >
                {ds.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <table className="benchmark-table">
        <thead>
          <tr>
            <th>Method</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Avg Latency</th>
            <th>Est. Cost / call</th>
          </tr>
        </thead>
        <tbody>
          {results.map(r => (
            <ResultRow key={r.method} r={r} />
          ))}
        </tbody>
      </table>

      {results.some(r => r.per_difficulty) && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '0.5rem', fontSize: '14px', fontWeight: 600 }}>
            F1 by Difficulty
          </h3>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
            {DIFFICULTY_ORDER.map(d => `n≈${DIFFICULTY_N[d]} ${d}`).join(' · ')}
          </p>
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Method</th>
                {DIFFICULTY_ORDER.map(d => <th key={d}>{d.charAt(0).toUpperCase() + d.slice(1)}</th>)}
              </tr>
            </thead>
            <tbody>
              {results.map(r => (
                <tr key={r.method}>
                  <td className="method-cell">{METHOD_LABELS[r.method] ?? r.method}</td>
                  {DIFFICULTY_ORDER.map(d => {
                    const f1 = r.per_difficulty?.[d]?.f1
                    return <td key={d} className="metric-cell">{f1 != null ? (f1 * 100).toFixed(1) + '%' : '-'}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
