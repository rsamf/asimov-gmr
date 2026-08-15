import { useEffect, useRef } from "react"
import type { Cell, Header, Row, RowData, Table } from "@tanstack/react-table"
import { useVirtualizer } from "@tanstack/react-virtual"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { formatValue, MISSING } from "@/lib/format"
import type { Enrichment } from "@/lib/enrichment"
import type { ClipRow, ColumnSpec, Difficulty } from "@/types"

declare module "@tanstack/react-table" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData extends RowData, TValue> {
    spec?: ColumnSpec
    enrichment?: Enrichment
  }
}

export const ROW_H = 30

export interface ScrollApi {
  scrollToIndex: (index: number) => void
}

interface ClipTableProps {
  table: Table<ClipRow>
  selected: string | null
  onSelect: (clipName: string) => void
  scrollApiRef: React.MutableRefObject<ScrollApi | null>
}

export function ClipTable({ table, selected, onSelect, scrollApiRef }: ClipTableProps) {
  const parentRef = useRef<HTMLDivElement>(null)
  const rows = table.getRowModel().rows

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_H,
    overscan: 10,
  })

  useEffect(() => {
    scrollApiRef.current = {
      scrollToIndex: (i) => virtualizer.scrollToIndex(i, { align: "auto" }),
    }
  })

  const items = virtualizer.getVirtualItems()
  const totalH = virtualizer.getTotalSize()
  const padTop = items.length ? items[0].start : 0
  const padBottom = items.length ? totalH - items[items.length - 1].end : 0
  const nCols = table.getVisibleLeafColumns().length

  return (
    <div
      ref={parentRef}
      className="min-h-0 flex-1 overflow-auto rounded-lg border border-border bg-card"
    >
      <table
        className="w-full table-fixed border-separate border-spacing-0 text-[13px]"
        style={{ minWidth: table.getTotalSize() }}
      >
        <colgroup>
          {table.getVisibleLeafColumns().map((c) => (
            <col key={c.id} style={{ width: c.getSize() }} />
          ))}
        </colgroup>
        <thead className="sticky top-0 z-10">
          <tr>
            {table.getHeaderGroups()[0].headers.map((h) => (
              <HeaderCell key={h.id} header={h} />
            ))}
          </tr>
        </thead>
        <tbody className="font-mono text-xs">
          {padTop > 0 && <SpacerRow height={padTop} colSpan={nCols} />}
          {items.map((vi) => {
            const row = rows[vi.index]
            return (
              <BodyRow
                key={row.id}
                row={row}
                isSelected={row.id === selected}
                onSelect={onSelect}
              />
            )
          })}
          {padBottom > 0 && <SpacerRow height={padBottom} colSpan={nCols} />}
          {rows.length === 0 && (
            <tr>
              <td colSpan={nCols} className="p-8 text-center font-sans text-sm text-muted-foreground">
                No clips match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

function SpacerRow({ height, colSpan }: { height: number; colSpan: number }) {
  return (
    <tr aria-hidden>
      <td colSpan={colSpan} style={{ height, padding: 0, border: 0 }} />
    </tr>
  )
}

function HeaderCell({ header }: { header: Header<ClipRow, unknown> }) {
  const col = header.column
  const meta = col.columnDef.meta
  const sorted = col.getIsSorted()
  const multi = header.getContext().table.getState().sorting.length > 1
  const align = meta?.enrichment?.align ?? "right"

  const label = (
    <button
      type="button"
      onClick={col.getToggleSortingHandler()}
      className={cn(
        "flex w-full items-center gap-1 truncate px-2.5 py-2 font-sans text-xs font-semibold hover:text-marker",
        align === "right" ? "justify-end" : "justify-start",
        sorted ? "text-marker" : "text-muted-foreground",
      )}
    >
      {col.getIsFiltered() && <span className="size-1.5 shrink-0 rounded-full bg-rec" title="filtered" />}
      <span className="truncate">{meta?.spec?.t ?? col.id}</span>
      {sorted && (
        <span className="shrink-0 text-[9px] leading-none">
          {sorted === "desc" ? "▼" : "▲"}
          {multi && <sup className="ml-px">{col.getSortIndex() + 1}</sup>}
        </span>
      )}
    </button>
  )

  return (
    <th className="border-b border-border bg-card p-0 font-normal">
      {meta?.spec?.d ? (
        <Tooltip>
          <TooltipTrigger asChild>{label}</TooltipTrigger>
          <TooltipContent side="bottom" align={align === "right" ? "end" : "start"} className="max-w-72 whitespace-normal text-left">
            {meta.spec.d}
          </TooltipContent>
        </Tooltip>
      ) : (
        label
      )}
    </th>
  )
}

function BodyRow({
  row,
  isSelected,
  onSelect,
}: {
  row: Row<ClipRow>
  isSelected: boolean
  onSelect: (clipName: string) => void
}) {
  return (
    <tr
      data-selected={isSelected || undefined}
      onClick={() => onSelect(row.id)}
      className="group/row cursor-pointer hover:bg-accent/40 data-selected:bg-accent"
      style={{ height: ROW_H }}
    >
      {row.getVisibleCells().map((cell, i) => (
        <CellTd key={cell.id} cell={cell} first={i === 0} />
      ))}
    </tr>
  )
}

function CellTd({ cell, first }: { cell: Cell<ClipRow, unknown>; first: boolean }) {
  const meta = cell.column.columnDef.meta
  const e = meta?.enrichment ?? {}
  const align = e.align ?? "right"
  return (
    <td
      className={cn(
        "truncate border-b border-border/40 px-2.5",
        align === "right" ? "text-right" : "text-left",
        first && "group-data-selected/row:shadow-[inset_2px_0_0_var(--marker)]",
      )}
    >
      <CellValue value={cell.getValue()} row={cell.row.original} spec={meta?.spec} enrichment={e} />
    </td>
  )
}

export const DIFFICULTY_DOT: Record<Difficulty, string> = {
  easy: "bg-easy",
  medium: "bg-medium",
  hard: "bg-hard",
}

function CellValue({
  value,
  row,
  spec,
  enrichment,
}: {
  value: unknown
  row: ClipRow
  spec?: ColumnSpec
  enrichment: Enrichment
}) {
  switch (enrichment.semantic) {
    case "clipname":
      return (
        <span className={row.has_video ? "text-foreground" : "text-muted-foreground"}>
          {String(value)}
          {!row.has_video && (
            <span className="ml-1.5 rounded-sm border border-rec/40 px-1 text-[10px] text-rec">
              no render
            </span>
          )}
        </span>
      )
    case "difficulty": {
      if (value == null) return <Missing />
      const d = value as Difficulty
      return (
        <span className="inline-flex items-center gap-1.5">
          <span className={cn("size-1.5 rounded-full", DIFFICULTY_DOT[d])} />
          {d}
        </span>
      )
    }
    case "passfail":
      if (value == null) return <Missing />
      return value === "succeeded" ? (
        <span className="text-pass">✓ {String(value)}</span>
      ) : (
        <span className="text-fail">✗ {String(value)}</span>
      )
    case "removed": {
      if (value == null) return <Missing />
      if (value === "") return <span className="text-muted-foreground/50">·</span>
      return (
        <span className={value === "error" ? "text-fail" : "text-muted-foreground"}>
          {String(value)}
        </span>
      )
    }
    case "split":
      if (value == null) return <Missing />
      return value === "test" ? (
        <span className="font-medium">test</span>
      ) : (
        <span className="text-muted-foreground">{String(value)}</span>
      )
    case "status": {
      if (value == null) return <Missing />
      const s = String(value)
      if (s === "ok") return <span className="text-muted-foreground">ok</span>
      return (
        <span className={s.startsWith("error") ? "text-fail" : "text-rec"} title={s}>
          {s}
        </span>
      )
    }
    case "bool":
      return value ? (
        <span className="text-fail">✓</span>
      ) : (
        <span className="text-muted-foreground/50">·</span>
      )
    default: {
      if (value == null) return <Missing />
      return <>{spec ? formatValue(spec.f, value) : String(value)}</>
    }
  }
}

function Missing() {
  return <span className="text-muted-foreground/50">{MISSING}</span>
}
