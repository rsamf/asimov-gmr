import { useEffect, useState } from "react"
import { fetchClips, fetchDatasets } from "@/lib/api"
import type { ClipsResponse, DatasetInfo } from "@/types"

// One payload per dataset per session (~2 MB each, rebuilt server-side on
// every request) — cache so switching back to a dataset is instant.
const cache = new Map<string, ClipsResponse>()

export interface ClipsState {
  data: ClipsResponse | null
  loading: boolean
  error: string | null
}

export function useClipsData(dataset: string): ClipsState {
  const [state, setState] = useState<ClipsState>(() => ({
    data: cache.get(dataset) ?? null,
    loading: !cache.has(dataset),
    error: null,
  }))

  useEffect(() => {
    const cached = cache.get(dataset)
    if (cached) {
      setState({ data: cached, loading: false, error: null })
      return
    }
    let alive = true
    setState({ data: null, loading: true, error: null })
    fetchClips(dataset)
      .then((d) => {
        cache.set(dataset, d)
        if (alive) setState({ data: d, loading: false, error: null })
      })
      .catch((e: unknown) => {
        if (alive) setState({ data: null, loading: false, error: String(e) })
      })
    return () => {
      alive = false
    }
  }, [dataset])

  return state
}

export function useDatasets(): DatasetInfo[] {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(console.error)
  }, [])
  return datasets
}
