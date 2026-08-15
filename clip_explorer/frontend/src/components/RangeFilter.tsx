import { useEffect, useState } from "react"
import { Input } from "@/components/ui/input"
import { binCounts, type ColumnStats } from "@/lib/stats"
import { HistogramBrush } from "./HistogramBrush"
import { PanelTitle } from "./FacetFilter"
import type { ColumnSpec } from "@/types"

interface RangeFilterPanelProps {
  col: ColumnSpec
  stats: ColumnStats
  /** rows passing every OTHER active filter (null = no other filters) */
  exceptMask: Uint8Array | null
  value: { lo: number | null; hi: number | null } | null
  unit?: string
  onChange: (lo: number | null, hi: number | null) => void
}

export function RangeFilterPanel({ col, stats, exceptMask, value, unit, onChange }: RangeFilterPanelProps) {
  const binsFiltered = exceptMask ? binCounts(stats, exceptMask) : null

  return (
    <div className="flex w-64 flex-col gap-1.5">
      <PanelTitle
        title={unit ? `${col.t} (${unit})` : col.t}
        onClear={value ? () => onChange(null, null) : undefined}
      />
      <HistogramBrush
        binsAll={stats.binsAll}
        binsFiltered={binsFiltered}
        min={stats.min}
        max={stats.max}
        width={256}
        height={72}
        range={value ? [value.lo, value.hi] : null}
        onRange={onChange}
      />
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>{trim(stats.min)}</span>
        <span>{trim(stats.max)}</span>
      </div>
      <div className="flex items-center gap-2">
        <BoundInput
          placeholder="min"
          bound={value?.lo ?? null}
          onCommit={(lo) => onChange(lo, value?.hi ?? null)}
        />
        <span className="text-xs text-muted-foreground">–</span>
        <BoundInput
          placeholder="max"
          bound={value?.hi ?? null}
          onCommit={(hi) => onChange(value?.lo ?? null, hi)}
        />
      </div>
      <p className="text-[10px] leading-snug text-muted-foreground">
        Brush the distribution or type bounds. Clips missing this metric are
        excluded while the filter is active.
      </p>
    </div>
  )
}

function BoundInput({
  placeholder,
  bound,
  onCommit,
}: {
  placeholder: string
  bound: number | null
  onCommit: (v: number | null) => void
}) {
  const [text, setText] = useState(bound == null ? "" : String(bound))
  useEffect(() => {
    setText(bound == null ? "" : String(bound))
  }, [bound])
  const commit = () => {
    const t = text.trim()
    const n = t === "" ? null : Number(t)
    onCommit(n != null && Number.isNaN(n) ? null : n)
  }
  return (
    <Input
      type="number"
      inputMode="decimal"
      placeholder={placeholder}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit()
      }}
      className="h-7 font-mono text-xs"
    />
  )
}

const trim = (v: number) => {
  const s = v.toFixed(2)
  return s.replace(/\.?0+$/, "")
}
