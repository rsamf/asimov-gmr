import type { ClipsResponse, DatasetInfo } from "@/types"

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const r = await fetch("/api/datasets")
  if (!r.ok) throw new Error(`GET /api/datasets failed (${r.status})`)
  return r.json()
}

export async function fetchClips(dataset: string): Promise<ClipsResponse> {
  const r = await fetch(`/api/clips?dataset=${encodeURIComponent(dataset)}`)
  if (!r.ok) throw new Error(`GET /api/clips failed (${r.status})`)
  return r.json()
}

/** Clip video paths contain spaces/commas — encode per path segment. */
export const encodeVideoPath = (name: string) =>
  name.split("/").map(encodeURIComponent).join("/")

export const videoUrl = (dataset: string, video: string) =>
  `/video/${encodeURIComponent(dataset)}/${encodeVideoPath(video)}`
