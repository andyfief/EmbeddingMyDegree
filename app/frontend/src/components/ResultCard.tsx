import { useState } from 'react'
import type { ChunkResult } from '../types'
import { openFile } from '../api'

interface Props {
  result: ChunkResult
  onOpened: (filePath: string) => void
}

function scoreColor(score: number): string {
  if (score >= 0.7) return '#4ade80'   // green
  if (score >= 0.5) return '#facc15'   // yellow
  return '#94a3b8'                      // grey
}

function shortPath(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/')
  return parts.slice(-3).join('/')
}

export default function ResultCard({ result, onOpened }: Props) {
  const [opening, setOpening] = useState(false)
  const [openError, setOpenError] = useState<string | null>(null)

  const handleOpen = async () => {
    setOpening(true)
    setOpenError(null)
    try {
      await openFile(result.file_path, result.start_page)
      onOpened(result.file_path)
    } catch (e: unknown) {
      setOpenError(e instanceof Error ? e.message : 'Failed to open')
    } finally {
      setOpening(false)
    }
  }

  const isPdf = result.file_path.toLowerCase().endsWith('.pdf')
  const openLabel = opening
    ? '…'
    : isPdf && result.start_page
      ? `↗ Open p.${result.start_page}`
      : '↗ Open'

  return (
    <div className="result-card">
      <div className="result-header">
        <span className="result-rank">#{result.rank}</span>
        <span className="result-score" style={{ color: scoreColor(result.score) }}>
          {result.score.toFixed(4)}
        </span>
        <span className="result-category">{result.category}</span>
        <span className="result-chunk">
          chunk {result.chunk_index + 1}/{result.total_chunks}
        </span>
        <button
          className="open-btn"
          onClick={handleOpen}
          disabled={opening}
          title={result.file_path}
        >
          {openLabel}
        </button>
      </div>

      <div className="result-path" title={result.file_path}>
        {shortPath(result.file_path)}
      </div>

      <div className="result-preview">{result.preview}</div>

      {openError && <div className="open-error">{openError}</div>}
    </div>
  )
}
