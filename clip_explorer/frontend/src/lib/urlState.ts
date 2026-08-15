import type { ColumnFiltersState, SortingState } from "@tanstack/react-table"
import type { ClipsResponse, SortSpec } from "@/types"
import { isNumericCol } from "./stats"
import type { FilterValue } from "./filters"

/**
 * Shareable views: ds / q / sort / f_<key> (facet) / r_<key> (range) / sel.
 * Facet tokens are individually URI-encoded so a token containing "," (e.g.
 * an error-message status) survives the comma-joined list.
 */

export interface UrlState {
  dataset: string | null
  q: string
  sorting: SortingState | null // null = unspecified → server default
  filters: ColumnFiltersState
  sel: string | null
}

export function parseUrl(search: string): UrlState {
  const p = new URLSearchParams(search)
  const state: UrlState = {
    dataset: p.get("ds"),
    q: p.get("q") ?? "",
    sorting: null,
    filters: [],
    sel: p.get("sel"),
  }
  const sort = p.get("sort")
  if (sort) {
    state.sorting = sort.split(",").flatMap((part) => {
      const m = /^(.+)\.(asc|desc)$/.exec(part)
      return m ? [{ id: m[1], desc: m[2] === "desc" }] : []
    })
  }
  for (const [k, v] of p) {
    if (k.startsWith("f_")) {
      const values = v.split(",").map((t) => {
        try {
          return decodeURIComponent(t)
        } catch {
          return t
        }
      })
      state.filters.push({ id: k.slice(2), value: { kind: "facet", values } })
    } else if (k.startsWith("r_")) {
      const m = /^(.*)\.\.(.*)$/.exec(v)
      if (!m) continue
      const lo = m[1] === "" ? null : Number(m[1])
      const hi = m[2] === "" ? null : Number(m[2])
      if ((lo != null && Number.isNaN(lo)) || (hi != null && Number.isNaN(hi))) continue
      state.filters.push({ id: k.slice(2), value: { kind: "range", lo, hi } })
    }
  }
  return state
}

export interface AppUrlState {
  dataset: string
  q: string
  sorting: SortingState
  filters: ColumnFiltersState
  sel: string | null
}

export function serializeUrl(state: AppUrlState, serverSort: SortSpec | null): string {
  const p = new URLSearchParams()
  p.set("ds", state.dataset)
  if (state.q.trim()) p.set("q", state.q)
  const def =
    serverSort && state.sorting.length === 1
      ? state.sorting[0].id === serverSort.key &&
        state.sorting[0].desc === (serverSort.dir === -1)
      : false
  if (state.sorting.length > 0 && !def) {
    p.set("sort", state.sorting.map((s) => `${s.id}.${s.desc ? "desc" : "asc"}`).join(","))
  }
  for (const f of state.filters) {
    const value = f.value as FilterValue
    if (value.kind === "facet") {
      p.set(`f_${f.id}`, value.values.map(encodeURIComponent).join(","))
    } else {
      p.set(`r_${f.id}`, `${value.lo ?? ""}..${value.hi ?? ""}`)
    }
  }
  if (state.sel) p.set("sel", state.sel)
  return `?${p.toString()}`
}

/**
 * Validate a parsed URL state against a loaded payload: drop filters/sorts
 * referencing columns this dataset doesn't have (or of the wrong kind), and
 * fall back to the server sort when nothing valid remains.
 */
export function applyToDataset(
  state: UrlState,
  resp: ClipsResponse,
): { sorting: SortingState; filters: ColumnFiltersState; sel: string | null } {
  const cols = new Map(resp.columns.map((c) => [c.k, c]))
  const filters = state.filters.filter((f) => {
    if (f.id === "has_video") return (f.value as FilterValue).kind === "facet"
    const col = cols.get(f.id)
    if (!col) return false
    const kind = (f.value as FilterValue).kind
    return kind === "range" ? isNumericCol(col) : !isNumericCol(col)
  })
  let sorting = (state.sorting ?? []).filter((s) => cols.has(s.id))
  if (sorting.length === 0) {
    sorting = [{ id: resp.sort.key, desc: resp.sort.dir === -1 }]
  }
  const sel =
    state.sel && resp.clips.some((c) => c.clip_name === state.sel) ? state.sel : null
  return { sorting, filters, sel }
}
