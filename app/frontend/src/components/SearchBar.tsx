import React from 'react'

interface Props {
  query: string
  k: number
  loading: boolean
  onQueryChange: (v: string) => void
  onKChange: (v: number) => void
  onSearch: () => void
}

export default function SearchBar({ query, k, loading, onQueryChange, onKChange, onSearch }: Props) {
  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim() && !loading) onSearch()
  }

  return (
    <div className="search-bar">
      <input
        className="search-input"
        type="text"
        placeholder="Search files..."
        value={query}
        onChange={e => onQueryChange(e.target.value)}
        onKeyDown={handleKey}
        autoFocus
      />
      <div className="search-controls">
        <label className="k-label">
          Top&nbsp;
          <input
            className="k-input"
            type="number"
            min={1}
            max={50}
            value={k}
            onChange={e => onKChange(Math.max(1, Math.min(50, Number(e.target.value))))}
          />
          &nbsp;results
        </label>
        <button
          className="search-btn"
          onClick={onSearch}
          disabled={loading || !query.trim()}
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </div>
    </div>
  )
}
