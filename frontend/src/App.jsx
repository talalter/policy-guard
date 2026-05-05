import { useState } from 'react'
import InputPanel from './components/InputPanel.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import BenchmarkTab from './components/BenchmarkTab.jsx'

export default function App() {
  const [activeTab, setActiveTab] = useState('checker')
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Contradiction Detector</h1>
        <nav className="tab-nav">
          <button
            className={activeTab === 'checker' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('checker')}
          >
            Checker
          </button>
          <button
            className={activeTab === 'benchmark' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('benchmark')}
          >
            Benchmark
          </button>
        </nav>
      </header>

      {activeTab === 'checker' ? (
        <div className="checker-layout">
          <InputPanel onReport={setReport} onLoading={setLoading} />
          <ResultPanel report={report} loading={loading} />
        </div>
      ) : (
        <BenchmarkTab />
      )}
    </div>
  )
}
