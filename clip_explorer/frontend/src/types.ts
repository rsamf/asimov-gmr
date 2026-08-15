/** Server contract for /api/datasets and /api/clips (clip_explorer/app.py). */

export type FormatHint = "text" | "int" | "f1" | "f2" | "f3" | "bool"

/** k = row key, t = header title, f = format hint, d = hover documentation. */
export interface ColumnSpec {
  k: string
  t: string
  f: FormatHint
  d: string
}

export interface DatasetInfo {
  name: string
  label: string
}

export type Difficulty = "easy" | "medium" | "hard"
export type Removed = "" | "glitch" | "rejected" | "fallen" | "short" | "error" | "excluded"

/**
 * One clip. Only identity keys are guaranteed; every metric key can be null
 * or absent entirely (e.g. no training CSV matched), so generic access goes
 * through numVal/strVal in lib/stats.ts rather than direct indexing.
 */
export interface ClipRow {
  clip_name: string
  dataset: string
  video: string
  has_video: boolean
  status?: string
  glitch?: boolean
  difficulty?: Difficulty | null
  driver?: string | null
  split?: "test" | "train"
  removed?: Removed
  pass?: "succeeded" | "failed" | null
  frames?: number
  duration_s?: number
  src?: string
  [key: string]: unknown
}

export interface SortSpec {
  key: string
  dir: 1 | -1
}

export interface ClipsResponse {
  dataset: string
  clips: ClipRow[]
  datasets: string[]
  columns: ColumnSpec[]
  sort: SortSpec
  total: number
  rendered: number
}
