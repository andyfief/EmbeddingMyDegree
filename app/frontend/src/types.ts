export interface ChunkResult {
  rank: number
  score: number
  file_path: string
  chunk_index: number
  total_chunks: number
  category: string
  preview: string
  start_page: number | null
}

export interface SearchResponse {
  query: string
  k: number
  results: ChunkResult[]
  latency_ms: number
}