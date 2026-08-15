import type { FilterFn } from "@tanstack/react-table"
import type { ClipRow } from "@/types"

/** Facet token for null/absent values — selectable as "—" in the UI. */
export const NULL_TOKEN = "~"

/** Canonical string token for any cell value (facet membership + URL state). */
export function tokenOf(v: unknown): string {
  if (v == null) return NULL_TOKEN
  if (typeof v === "boolean") return v ? "1" : "0"
  return String(v)
}

export interface RangeFilterValue {
  kind: "range"
  lo: number | null
  hi: number | null
}

export interface FacetFilterValue {
  kind: "facet"
  values: string[]
}

export type FilterValue = RangeFilterValue | FacetFilterValue

/** Rows with a missing metric are excluded while a range filter is active. */
export function rowPassesRange(v: unknown, f: RangeFilterValue): boolean {
  if (typeof v !== "number" || Number.isNaN(v)) return false
  return (f.lo == null || v >= f.lo) && (f.hi == null || v <= f.hi)
}

export function rowPassesFacet(v: unknown, f: FacetFilterValue): boolean {
  return f.values.includes(tokenOf(v))
}

export function rowPassesFilter(v: unknown, f: FilterValue): boolean {
  return f.kind === "range" ? rowPassesRange(v, f) : rowPassesFacet(v, f)
}

/** Token-AND substring search over clip name + AMASS collection. */
export function rowMatchesQuery(row: ClipRow, q: string): boolean {
  const hay = `${row.clip_name} ${row.dataset}`.toLowerCase()
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((tok) => !tok || hay.includes(tok))
}

// --- TanStack filterFns (row.getValue goes through the null→undefined accessor) ---

export const numberRange: FilterFn<ClipRow> = (row, id, value: RangeFilterValue) =>
  rowPassesRange(row.getValue(id), value)

export const facetSet: FilterFn<ClipRow> = (row, id, value: FacetFilterValue) =>
  rowPassesFacet(row.getValue(id), value)

export const globalSearch: FilterFn<ClipRow> = (row, _id, q: string) =>
  rowMatchesQuery(row.original, q)
