import { useState } from 'react'
import type { ChunkResult, SearchResponse } from './types'
import { search as apiSearch } from './api'
import SearchBar from './components/SearchBar'
import ResultCard from './components/ResultCard'

export default function App() {
  const [query, setQuery]           = useState('')
  const [k, setK]                   = useState(5)
  const [loading, setLoading]       = useState(false)
  const [results, setResults]       = useState<ChunkResult[] | null>(null)
  const [error, setError]           = useState<string | null>(null)
  const [latencyMs, setLatencyMs]   = useState<number | null>(null)
  const [lastOpened, setLastOpened] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setError(null)
    setResults(null)
    setLastOpened(null)
    try {
      const resp: SearchResponse = await apiSearch(query.trim(), k)
      setResults(resp.results)
      setLatencyMs(resp.latency_ms)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleOpened = (filePath: string) => {
    setLastOpened(filePath)
    setTimeout(() => setLastOpened(null), 3000)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">📚 Embedding my CS Degree</h1>
        <p className="app-subtitle">Semantic search over my embedded school files</p>
      </header>

      <main className="app-main">
        <SearchBar
          query={query}
          k={k}
          loading={loading}
          onQueryChange={setQuery}
          onKChange={setK}
          onSearch={handleSearch}
        />

        {(results !== null || error || loading) && (
          <div className="status-bar">
            {loading && <span className="status-loading">Embedding query and searching…</span>}
            {error   && <span className="status-error">⚠ {error}</span>}
            {results !== null && !loading && (
              <span className="status-ok">
                {results.length} results &nbsp;·&nbsp; {latencyMs?.toFixed(0)} ms
              </span>
            )}
          </div>
        )}

        {lastOpened && (
          <div className="toast">
            ✓ Opened: {lastOpened.split('/').slice(-1)[0]}
          </div>
        )}

        {results !== null && !loading && (
          <div className="results-list">
            {results.length === 0
              ? <p className="no-results">No results found.</p>
              : results.map(r => (
                  <ResultCard key={`${r.file_path}-${r.chunk_index}`} result={r} onOpened={handleOpened} />
                ))
            }
          </div>
        )}
      </main>
    </div>
  )
}
