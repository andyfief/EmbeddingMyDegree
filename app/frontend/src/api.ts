import type { SearchResponse } from './types'

const BASE = 'http://localhost:8000'

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {}
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function search(query: string, k: number): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k }),
  })
  return handleResponse<SearchResponse>(res)
}

export async function openFile(filePath: string, startPage?: number | null): Promise<void> {
  const res = await fetch(`${BASE}/open`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_path: filePath, start_page: startPage ?? null }),
  })
  await handleResponse<unknown>(res)
}