import type { ClipRow, ClipsResponse, ColumnSpec } from "@/types"
import { rowMatchesQuery, rowPassesFilter, tokenOf, type FilterValue } from "./filters"

/** Typed access into the server-driven row dict. */
export function numVal(row: ClipRow, key: string): number | null {
  const v = row[key]
  return typeof v === "number" && Number.isFinite(v) ? v : null
}

export function strVal(row: ClipRow, key: string): string | null {
  const v = row[key]
  return typeof v === "string" ? v : null
}

const NUMERIC_HINTS = new Set(["int", "f1", "f2", "f3"])
export const isNumericCol = (c: ColumnSpec) => NUMERIC_HINTS.has(c.f)

export const N_BINS = 30

export interface ColumnStats {
  key: string
  values: Float64Array // aligned with resp.clips; NaN = missing
  count: number // non-missing values
  min: number
  max: number
  binWidth: number
  binsAll: Uint32Array // histogram over ALL rows — the bin domain is fixed
}

const statsCache = new WeakMap<ClipsResponse, Map<string, ColumnStats>>()

/** Per-numeric-column value arrays + fixed-domain histograms, computed once per payload. */
export function columnStats(resp: ClipsResponse): Map<string, ColumnStats> {
  const hit = statsCache.get(resp)
  if (hit) return hit
  const m = new Map<string, ColumnStats>()
  const n = resp.clips.length
  for (const col of resp.columns) {
    if (!isNumericCol(col)) continue
    const values = new Float64Array(n)
    let min = Infinity
    let max = -Infinity
    let count = 0
    for (let i = 0; i < n; i++) {
      const v = numVal(resp.clips[i], col.k)
      if (v == null) {
        values[i] = NaN
        continue
      }
      values[i] = v
      count++
      if (v < min) min = v
      if (v > max) max = v
    }
    if (count === 0) {
      min = 0
      max = 1
    } else if (min === max) {
      max = min + 1 // degenerate distribution still gets a drawable domain
    }
    const binWidth = (max - min) / N_BINS
    const binsAll = new Uint32Array(N_BINS)
    for (let i = 0; i < n; i++) {
      if (!Number.isNaN(values[i])) binsAll[binIndex(values[i], min, binWidth)]++
    }
    m.set(col.k, { key: col.k, values, count, min, max, binWidth, binsAll })
  }
  statsCache.set(resp, m)
  return m
}

export function binIndex(v: number, min: number, binWidth: number): number {
  const i = Math.floor((v - min) / binWidth)
  return i < 0 ? 0 : i >= N_BINS ? N_BINS - 1 : i
}

/** Histogram of a column over a row subset (null mask = all rows). */
export function binCounts(stats: ColumnStats, mask: Uint8Array | null): Uint32Array {
  if (!mask) return stats.binsAll
  const bins = new Uint32Array(N_BINS)
  const { values, min, binWidth } = stats
  for (let i = 0; i < values.length; i++) {
    if (mask[i] && !Number.isNaN(values[i])) bins[binIndex(values[i], min, binWidth)]++
  }
  return bins
}

/** Share (0–100) of non-missing values ≤ v. */
export function percentile(stats: ColumnStats, v: number): number {
  let below = 0
  for (let i = 0; i < stats.values.length; i++) {
    const x = stats.values[i]
    if (!Number.isNaN(x) && x <= v) below++
  }
  return stats.count ? (100 * below) / stats.count : 0
}

// --- Cross-filter machinery -------------------------------------------------
//
// Facet counts and filter histograms must reflect every OTHER active filter
// but not the column's own (else selecting a facet value zeroes its siblings).
// TanStack's built-in faceting includes the column's own filter, so we keep
// one pass/fail mask per active filter and AND them with one exclusion.

export interface ActiveColumnFilter {
  id: string
  value: FilterValue
}

export const GLOBAL_MASK_ID = "__global__"

export function computeMasks(
  resp: ClipsResponse,
  columnFilters: ActiveColumnFilter[],
  globalFilter: string,
): Map<string, Uint8Array> {
  const n = resp.clips.length
  const masks = new Map<string, Uint8Array>()
  for (const f of columnFilters) {
    const mask = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      mask[i] = rowPassesFilter(resp.clips[i][f.id], f.value) ? 1 : 0
    }
    masks.set(f.id, mask)
  }
  const q = globalFilter.trim()
  if (q) {
    const mask = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      mask[i] = rowMatchesQuery(resp.clips[i], q) ? 1 : 0
    }
    masks.set(GLOBAL_MASK_ID, mask)
  }
  return masks
}

/** AND of all masks except `exceptId`; null = nothing active → all rows pass. */
export function maskExcept(
  masks: Map<string, Uint8Array>,
  n: number,
  exceptId: string | null,
): Uint8Array | null {
  let out: Uint8Array | null = null
  for (const [id, m] of masks) {
    if (id === exceptId) continue
    if (!out) {
      out = m.slice()
    } else {
      for (let i = 0; i < n; i++) if (!m[i]) out[i] = 0
    }
  }
  return out
}

/** Token → row count for a column over a subset (null mask = all rows). */
export function facetCounts(
  resp: ClipsResponse,
  key: string,
  mask: Uint8Array | null,
): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i < resp.clips.length; i++) {
    if (mask && !mask[i]) continue
    const t = tokenOf(resp.clips[i][key])
    counts.set(t, (counts.get(t) ?? 0) + 1)
  }
  return counts
}
