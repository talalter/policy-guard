/**
 * API client - centralises all fetch calls so components stay presentation-only.
 *
 * Every function throws on non-2xx responses; callers handle errors in try/catch.
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function getSessionId() {
  let id = localStorage.getItem('pg_session_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('pg_session_id', id)
  }
  return id
}

async function request(path, options = {}) {
  const { headers: extraHeaders, ...rest } = options
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      'X-Session-ID': getSessionId(),
      ...extraHeaders,
    },
    ...rest,
  })
  if (!res.ok) throw new Error(`API returned ${res.status}`)
  if (res.status === 204) return null
  return res.json()
}

export function checkContradictions(context, response) {
  return request('/check', {
    method: 'POST',
    body: JSON.stringify({ context, response }),
  })
}

export function checkLlmOnly(context, response) {
  return request('/check/llm-only', {
    method: 'POST',
    body: JSON.stringify({ context, response }),
  })
}

export function submitFeedback(runId, violationIndex, verdict) {
  return request(`/feedback/${runId}`, {
    method: 'POST',
    body: JSON.stringify({ violation_index: violationIndex, verdict }),
  })
}

export function fetchBenchmarkDatasets() {
  return request('/benchmark-datasets')
}

export function fetchBenchmarkResults(datasetKey) {
  const qs = datasetKey ? `?dataset=${encodeURIComponent(datasetKey)}` : ''
  return request(`/benchmark-results${qs}`)
}

export function fetchHistory() {
  return request('/history')
}

export function fetchHistoryItem(runId) {
  return request(`/history/${runId}`)
}

export function deleteHistoryItem(runId) {
  return request(`/history/${runId}`, { method: 'DELETE' })
}

export function fetchStats() {
  return request('/stats')
}
