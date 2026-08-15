import { useEffect, useMemo, useState } from "react"
import type { ColumnFiltersState } from "@tanstack/react-table"
import { ChevronLeft, ListFilter, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { columnStats, isNumericCol, maskExcept } from "@/lib/stats"
import { enrichmentOf, facetLabel, isFacetable } from "@/lib/enrichment"
import type { FacetFilterValue, FilterValue, RangeFilterValue } from "@/lib/filters"
import { FacetFilterPanel } from "./FacetFilter"
import { RangeFilterPanel } from "./RangeFilter"
import type { ClipsResponse, ColumnSpec } from "@/types"

/** Pseudo-column: has_video is row data but not a server table column. */
const HAS_VIDEO_COL: ColumnSpec = {
  k: "has_video",
  t: "has render",
  f: "bool",
  d: "Whether a review MP4 has been rendered for this clip.",
}

interface FilterBarProps {
  resp: ClipsResponse
  columnFilters: ColumnFiltersState
  setColumnFilters: React.Dispatch<React.SetStateAction<ColumnFiltersState>>
  globalFilter: string
  setGlobalFilter: (q: string) => void
  masks: Map<string, Uint8Array>
  searchRef: React.RefObject<HTMLInputElement | null>
  viewCount: number
}

export function FilterBar({
  resp,
  columnFilters,
  setColumnFilters,
  globalFilter,
  setGlobalFilter,
  masks,
  searchRef,
  viewCount,
}: FilterBarProps) {
  // local echo of the search box, debounced up into table state
  const [q, setQ] = useState(globalFilter)
  useEffect(() => setQ(globalFilter), [globalFilter])
  useEffect(() => {
    const t = setTimeout(() => {
      if (q !== globalFilter) setGlobalFilter(q)
    }, 150)
    return () => clearTimeout(t)
  }, [q, globalFilter, setGlobalFilter])

  const [addOpen, setAddOpen] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)

  const { facetCols, rangeCols, colByKey } = useMemo(() => {
    const facetCols = [HAS_VIDEO_COL, ...resp.columns.filter((c) => isFacetable(c, resp))]
    const rangeCols = resp.columns.filter(isNumericCol)
    const colByKey = new Map(facetCols.concat(rangeCols).map((c) => [c.k, c]))
    return { facetCols, rangeCols, colByKey }
  }, [resp])

  const filterValue = (key: string): FilterValue | null =>
    (columnFilters.find((f) => f.id === key)?.value as FilterValue | undefined) ?? null

  const setFilter = (key: string, value: FilterValue | null) => {
    setColumnFilters((old) => {
      const rest = old.filter((f) => f.id !== key)
      return value == null ? rest : [...rest, { id: key, value }]
    })
  }

  const notRendered = resp.total - resp.rendered

  const editorFor = (key: string) => {
    const col = colByKey.get(key)
    if (!col) return null
    const exceptMask = maskExcept(masks, resp.clips.length, key)
    if (isNumericCol(col)) {
      const stats = columnStats(resp).get(key)
      if (!stats) return null
      const v = filterValue(key) as RangeFilterValue | null
      return (
        <RangeFilterPanel
          col={col}
          stats={stats}
          exceptMask={exceptMask}
          value={v ? { lo: v.lo, hi: v.hi } : null}
          unit={enrichmentOf(key).unit}
          onChange={(lo, hi) =>
            setFilter(key, lo == null && hi == null ? null : { kind: "range", lo, hi })
          }
        />
      )
    }
    const v = filterValue(key) as FacetFilterValue | null
    return (
      <FacetFilterPanel
        col={col}
        resp={resp}
        exceptMask={exceptMask}
        value={v?.values ?? null}
        onChange={(tokens) =>
          setFilter(key, tokens == null ? null : { kind: "facet", values: tokens })
        }
      />
    )
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Input
        ref={searchRef}
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") e.currentTarget.blur()
        }}
        placeholder="search clips…  /"
        className="h-7 w-56 text-xs"
        aria-label="Search clips"
      />

      <Popover
        open={addOpen}
        onOpenChange={(o) => {
          setAddOpen(o)
          if (!o) setEditing(null)
        }}
      >
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm">
            <ListFilter />
            filter
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-auto p-2.5">
          {editing == null ? (
            <div className="grid w-72 grid-cols-2 gap-x-3">
              <ColumnList title="categories" cols={facetCols} onPick={setEditing} />
              <ColumnList title="metric ranges" cols={rangeCols} onPick={setEditing} />
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <Button
                variant="ghost"
                size="xs"
                className="w-fit -translate-x-1 text-muted-foreground"
                onClick={() => setEditing(null)}
              >
                <ChevronLeft /> all filters
              </Button>
              {editorFor(editing)}
            </div>
          )}
        </PopoverContent>
      </Popover>

      {columnFilters.map((f) => {
        const col = colByKey.get(f.id)
        if (!col) return null
        return (
          <Popover key={f.id}>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="flex h-7 items-center gap-1 rounded-md border border-rec/40 bg-rec/10 px-2 font-mono text-[11px] text-foreground hover:border-rec/70"
              >
                {chipText(col, f.value as FilterValue)}
                <X
                  className="size-3 text-muted-foreground hover:text-foreground"
                  onClick={(e) => {
                    e.stopPropagation()
                    setFilter(f.id, null)
                  }}
                />
              </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-auto p-2.5">
              {editorFor(f.id)}
            </PopoverContent>
          </Popover>
        )
      })}

      {(columnFilters.length > 0 || globalFilter) && (
        <Button
          variant="ghost"
          size="xs"
          className="text-muted-foreground"
          onClick={() => {
            setColumnFilters([])
            setGlobalFilter("")
          }}
        >
          clear all
        </Button>
      )}

      <span className="ml-auto whitespace-nowrap font-mono text-[11px] text-muted-foreground">
        {viewCount.toLocaleString()} / {resp.total.toLocaleString()} clips
        {notRendered > 0 && ` · ${notRendered.toLocaleString()} not rendered`}
      </span>
    </div>
  )
}

function ColumnList({
  title,
  cols,
  onPick,
}: {
  title: string
  cols: ColumnSpec[]
  onPick: (key: string) => void
}) {
  return (
    <div className="flex flex-col">
      <span className="mb-1 px-1.5 font-condensed text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {title}
      </span>
      {cols.map((c) => (
        <button
          key={c.k}
          type="button"
          className="rounded-sm px-1.5 py-1 text-left text-xs hover:bg-accent"
          onClick={() => onPick(c.k)}
        >
          {c.t}
        </button>
      ))}
    </div>
  )
}

function chipText(col: ColumnSpec, value: FilterValue): string {
  if (value.kind === "range") {
    const lo = value.lo != null ? fmtNum(value.lo) : null
    const hi = value.hi != null ? fmtNum(value.hi) : null
    if (lo != null && hi != null) return `${col.t} ${lo}–${hi}`
    if (lo != null) return `${col.t} ≥ ${lo}`
    if (hi != null) return `${col.t} ≤ ${hi}`
    return col.t
  }
  const labels = value.values.map((t) => facetLabel(col.k, t))
  const shown = labels.slice(0, 2).join(", ")
  return `${col.t}: ${shown}${labels.length > 2 ? ` +${labels.length - 2}` : ""}`
}

const fmtNum = (v: number) => {
  const s = v.toFixed(2)
  return s.replace(/\.?0+$/, "")
}
