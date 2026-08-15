import type { ClipsResponse, ColumnSpec } from "@/types"
import { isNumericCol } from "./stats"
import { tokenOf } from "./filters"

/**
 * Client-side knowledge about known column keys. The server's column list is
 * the source of truth for presence/order/tooltips — a column missing from
 * this map still renders generically via its format hint, so adding a column
 * in app.py just appears. Entries here only add semantics.
 */

export type DetailGroup = "ik" | "dynamics" | "training" | "provenance" | "other"

export const GROUP_LABELS: Record<DetailGroup, string> = {
  ik: "IK quality",
  dynamics: "Dynamics",
  training: "Training feedback",
  provenance: "Provenance",
  other: "Other",
}

export type Semantic =
  | "clipname"
  | "difficulty"
  | "passfail"
  | "removed"
  | "split"
  | "status"
  | "bool"

export interface Enrichment {
  unit?: string
  group?: DetailGroup
  /** show a distribution sparkline for this metric in the detail pane */
  histogram?: boolean
  semantic?: Semantic
  align?: "left" | "right"
  width?: number
}

export const ENRICHMENT: Record<string, Enrichment> = {
  clip_name: { group: "provenance", semantic: "clipname", align: "left", width: 330 },
  dataset: { group: "provenance", align: "left", width: 120 },
  sat: { unit: "%", group: "ik", histogram: true, width: 62 },
  ankle: { unit: "%", group: "ik", histogram: true, width: 64 },
  pos_err: { unit: "cm", group: "ik", histogram: true, width: 72 },
  foot: { unit: "cm", group: "ik", histogram: true, width: 60 },
  wrist: { unit: "cm", group: "ik", histogram: true, width: 62 },
  float: { unit: "%", group: "ik", histogram: true, width: 60 },
  peak_v: { unit: "°/fr", group: "ik", histogram: true, width: 70 },
  glitch: { group: "ik", semantic: "bool", width: 58 },
  root_v: { unit: "m/s", group: "dynamics", histogram: true, width: 66 },
  root_av: { unit: "°/s", group: "dynamics", histogram: true, width: 70 },
  tilt: { unit: "°", group: "dynamics", histogram: true, width: 58 },
  difficulty: { group: "dynamics", semantic: "difficulty", width: 86 },
  split: { group: "training", semantic: "split", width: 62 },
  status: { group: "provenance", semantic: "status", align: "left", width: 74 },
  removed: { group: "provenance", semantic: "removed", align: "left", width: 88 },
  pass: { group: "training", semantic: "passfail", width: 92 },
  trained: { group: "training", align: "left", width: 70 },
  succ: { group: "training", histogram: true, width: 56 },
  mpkpe_g: { unit: "mm", group: "training", histogram: true, width: 76 },
  mpkpe_r: { unit: "mm", group: "training", histogram: true, width: 76 },
  // detail-pane-only keys (not in the server column list)
  driver: { group: "dynamics" },
  frames: { group: "provenance" },
  duration_s: { unit: "s", group: "provenance" },
  src: { group: "provenance" },
  video: { group: "provenance" },
  succ_passes: { group: "training" },
  n_passes: { group: "training" },
  last_kpe_mm: { unit: "mm", group: "training" },
}

export const enrichmentOf = (key: string): Enrichment => ENRICHMENT[key] ?? {}

/** Human label for a facet token of a given column. */
export function facetLabel(key: string, token: string): string {
  if (token === "~") return "—"
  if (key === "removed" && token === "") return "in training set"
  if (key === "glitch" || key === "has_video") return token === "1" ? "yes" : "no"
  return token
}

const MAX_FACET_VALUES = 25

/**
 * Which columns get facet-filter UI: known categoricals plus any future
 * non-numeric column with few distinct values. clip_name is search territory.
 */
export function isFacetable(col: ColumnSpec, resp: ClipsResponse): boolean {
  if (col.k === "clip_name") return false
  if (col.f === "bool") return true
  if (isNumericCol(col)) return false
  const seen = new Set<string>()
  for (const row of resp.clips) {
    seen.add(tokenOf(row[col.k]))
    if (seen.size > MAX_FACET_VALUES) return false
  }
  return true
}
