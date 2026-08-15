import { useEffect, useMemo, useRef, useState } from "react"
import {
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type ColumnFiltersState,
  type SortingFn,
  type SortingState,
  type VisibilityState,
} from "@tanstack/react-table"
import { TooltipProvider } from "@/components/ui/tooltip"
import { ClipTable, type ScrollApi } from "@/components/ClipTable"
import { DetailPane } from "@/components/DetailPane"
import { FilterBar } from "@/components/FilterBar"
import { ShortcutsHelp } from "@/components/ShortcutsHelp"
import { TopBar } from "@/components/TopBar"
import { useClipsData, useDatasets } from "@/hooks/useClipsData"
import { useKeyboard } from "@/hooks/useKeyboard"
import { videoUrl } from "@/lib/api"
import { enrichmentOf } from "@/lib/enrichment"
import { facetSet, globalSearch, numberRange } from "@/lib/filters"
import { computeMasks, isNumericCol, type ActiveColumnFilter } from "@/lib/stats"
import { applyToDataset, parseUrl, serializeUrl, type UrlState } from "@/lib/urlState"
import type { ClipRow, ClipsResponse } from "@/types"
import type { Preset } from "@/lib/presets"

const SEVERITY: Record<string, number> = { easy: 0, medium: 1, hard: 2 }

const difficultySort: SortingFn<ClipRow> = (a, b, id) =>
  (SEVERITY[a.getValue<string>(id)] ?? -1) - (SEVERITY[b.getValue<string>(id)] ?? -1)

const EMPTY_ROWS: ClipRow[] = []
const VISIBILITY_KEY = "clip-explorer.columnVisibility"

function loadVisibility(): VisibilityState {
  try {
    return JSON.parse(localStorage.getItem(VISIBILITY_KEY) ?? "{}") as VisibilityState
  } catch {
    return {}
  }
}

function buildColumnDefs(data: ClipsResponse | null): ColumnDef<ClipRow, unknown>[] {
  if (!data) return []
  const defs: ColumnDef<ClipRow, unknown>[] = data.columns.map((col) => {
    const e = enrichmentOf(col.k)
    const numeric = isNumericCol(col)
    return {
      id: col.k,
      // null → undefined so sortUndefined:"last" catches both null and absent
      accessorFn: (row) => {
        const v = row[col.k]
        return v == null ? undefined : v
      },
      meta: { spec: col, enrichment: e },
      size: e.width ?? (numeric ? 68 : 100),
      sortDescFirst: numeric || col.k === "difficulty",
      sortUndefined: "last",
      sortingFn: col.k === "difficulty" ? difficultySort : numeric ? "basic" : "text",
      filterFn: numeric ? numberRange : facetSet,
      enableGlobalFilter: col.k === "clip_name",
    }
  })
  // hidden pseudo-column so has_video is filterable like everything else
  defs.push({
    id: "has_video",
    accessorFn: (row) => row.has_video,
    enableSorting: false,
    enableGlobalFilter: false,
    filterFn: facetSet,
  })
  return defs
}

export default function App() {
  const initialUrl = useRef<UrlState>(parseUrl(window.location.search))
  const [dataset, setDataset] = useState(() => initialUrl.current.dataset ?? "v8")
  const datasets = useDatasets()
  const { data, loading, error } = useClipsData(dataset)

  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState("")
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(loadVisibility)
  const [selected, setSelected] = useState<string | null>(null)
  const [helpOpen, setHelpOpen] = useState(false)

  const pendingUrl = useRef<UrlState | null>(initialUrl.current)
  const pushNext = useRef(false)
  const scrollToSelRef = useRef(false)
  const searchRef = useRef<HTMLInputElement | null>(null)
  const scrollApiRef = useRef<ScrollApi | null>(null)
  const respRef = useRef<ClipsResponse | null>(null)

  const columnDefs = useMemo(() => buildColumnDefs(data), [data])
  const visibility = useMemo<VisibilityState>(
    () => ({ ...columnVisibility, has_video: false }),
    [columnVisibility],
  )

  const table = useReactTable({
    data: data?.clips ?? EMPTY_ROWS,
    columns: columnDefs,
    state: { sorting, columnFilters, globalFilter, columnVisibility: visibility },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    globalFilterFn: globalSearch,
    getRowId: (row) => row.clip_name,
    enableMultiSort: true,
    maxMultiSortColCount: 3,
    isMultiSortEvent: (e) => (e as MouseEvent).shiftKey,
  })

  // a payload arrived: restore the pending URL view, or reset to defaults
  useEffect(() => {
    if (!data || respRef.current === data) return
    respRef.current = data
    const pending = pendingUrl.current
    pendingUrl.current = null
    if (pending) {
      const applied = applyToDataset(pending, data)
      setSorting(applied.sorting)
      setColumnFilters(applied.filters)
      setGlobalFilter(pending.q)
      setSelected(applied.sel)
      if (applied.sel) scrollToSelRef.current = true
    } else {
      setSorting([{ id: data.sort.key, desc: data.sort.dir === -1 }])
      setColumnFilters([])
      setGlobalFilter("")
      setSelected(null)
    }
  }, [data])

  // keep a valid selection as filters/sort reshape the view
  const rows = table.getRowModel().rows
  useEffect(() => {
    if (!rows.length) {
      if (selected !== null) setSelected(null)
      return
    }
    if (!selected || !rows.some((r) => r.id === selected)) {
      setSelected(rows[0].id)
      return
    }
    if (scrollToSelRef.current) {
      // a URL-restored selection should be visible, not stranded off-screen
      scrollToSelRef.current = false
      const i = rows.findIndex((r) => r.id === selected)
      if (i >= 0) scrollApiRef.current?.scrollToIndex(i)
    }
  }, [rows, selected])

  // URL sync: replaceState while editing; pushState on dataset/preset switches
  useEffect(() => {
    if (!data) return
    const t = setTimeout(() => {
      const url = serializeUrl(
        { dataset, q: globalFilter, sorting, filters: columnFilters, sel: selected },
        data.sort,
      )
      if (url !== window.location.search) {
        if (pushNext.current) history.pushState(null, "", url)
        else history.replaceState(null, "", url)
      }
      pushNext.current = false
    }, 250)
    return () => clearTimeout(t)
  }, [data, dataset, sorting, columnFilters, globalFilter, selected])

  useEffect(() => {
    const onPop = () => {
      const parsed = parseUrl(window.location.search)
      const ds = parsed.dataset ?? "v8"
      if (ds === dataset && respRef.current) {
        const applied = applyToDataset(parsed, respRef.current)
        setSorting(applied.sorting)
        setColumnFilters(applied.filters)
        setGlobalFilter(parsed.q)
        setSelected(applied.sel)
      } else {
        pendingUrl.current = parsed
        setDataset(ds)
      }
    }
    window.addEventListener("popstate", onPop)
    return () => window.removeEventListener("popstate", onPop)
  }, [dataset])

  useEffect(() => {
    localStorage.setItem(VISIBILITY_KEY, JSON.stringify(columnVisibility))
  }, [columnVisibility])

  const masks = useMemo(
    () =>
      data
        ? computeMasks(data, columnFilters as ActiveColumnFilter[], globalFilter)
        : new Map<string, Uint8Array>(),
    [data, columnFilters, globalFilter],
  )

  const selectedClip = useMemo(
    () => data?.clips.find((c) => c.clip_name === selected) ?? null,
    [data, selected],
  )

  useKeyboard(
    useMemo(
      () => ({
        onMove: (delta: number) => {
          const list = table.getRowModel().rows
          if (!list.length) return
          const i = selected ? list.findIndex((r) => r.id === selected) : -1
          const next = Math.min(list.length - 1, Math.max(0, i + delta))
          setSelected(list[next].id)
          scrollApiRef.current?.scrollToIndex(next)
        },
        onTogglePlay: () => {
          const v = document.getElementById("clip-video") as HTMLVideoElement | null
          if (!v?.getAttribute("src")) return
          if (v.paused) void v.play()
          else v.pause()
        },
        onOpen: () => {
          if (selectedClip?.has_video) {
            window.open(videoUrl(dataset, selectedClip.video), "_blank")
          }
        },
        onFocusSearch: () => searchRef.current?.focus(),
        onHelp: () => setHelpOpen((o) => !o),
      }),
      [table, selected, selectedClip, dataset],
    ),
  )

  const applyPreset = (p: Preset) => {
    pushNext.current = true
    setColumnFilters(p.filters)
    setSorting(p.sorting)
    setGlobalFilter("")
  }

  const switchDataset = (name: string) => {
    if (name === dataset) return
    pushNext.current = true
    setDataset(name)
  }

  return (
    <TooltipProvider delayDuration={350}>
      <div className="flex h-screen flex-col">
        <TopBar
          datasets={datasets}
          dataset={dataset}
          onDatasetChange={switchDataset}
          onApplyPreset={applyPreset}
          table={table}
          onShowHelp={() => setHelpOpen(true)}
        />
        <div className="flex min-h-0 flex-1">
          <main className="flex min-w-0 flex-1 flex-col gap-2 p-3">
            {data && (
              <FilterBar
                resp={data}
                columnFilters={columnFilters}
                setColumnFilters={setColumnFilters}
                globalFilter={globalFilter}
                setGlobalFilter={setGlobalFilter}
                masks={masks}
                searchRef={searchRef}
                viewCount={rows.length}
              />
            )}
            {error ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
                <span className="text-fail">failed to load {dataset}</span>
                <span className="font-mono text-xs">{error}</span>
                <span>is the Flask app running on :5001?</span>
              </div>
            ) : loading ? (
              <div className="flex flex-1 items-center justify-center font-mono text-xs text-muted-foreground">
                loading {dataset}…
              </div>
            ) : (
              <ClipTable
                table={table}
                selected={selected}
                onSelect={setSelected}
                scrollApiRef={scrollApiRef}
              />
            )}
          </main>
          <aside className="flex w-[400px] shrink-0 flex-col border-l border-border bg-card/50">
            {data && <DetailPane clip={selectedClip} resp={data} dataset={dataset} />}
          </aside>
        </div>
        <ShortcutsHelp open={helpOpen} onOpenChange={setHelpOpen} />
      </div>
    </TooltipProvider>
  )
}
