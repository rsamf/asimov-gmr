import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { HistogramBrush } from "./HistogramBrush"
import { percentile, type ColumnStats } from "@/lib/stats"

export function MetricSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-0.5">
      <h3 className="mb-1 font-condensed text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {title}
      </h3>
      {children}
    </section>
  )
}

interface MetricRowProps {
  label: string
  /** hover documentation (the server column tooltip) */
  doc?: string
  value: React.ReactNode
  unit?: string
  /** when set (with a numeric value), draws the corpus distribution + marker */
  stats?: ColumnStats
  numericValue?: number | null
}

export function MetricRow({ label, doc, value, unit, stats, numericValue }: MetricRowProps) {
  const labelEl = (
    <span className="truncate text-muted-foreground">{label}</span>
  )
  return (
    <div className="flex h-6 items-center justify-between gap-3 text-xs">
      {doc ? (
        <Tooltip>
          <TooltipTrigger asChild>{labelEl}</TooltipTrigger>
          <TooltipContent side="left" className="max-w-72 whitespace-normal">
            {doc}
          </TooltipContent>
        </Tooltip>
      ) : (
        labelEl
      )}
      <span className="flex min-w-0 items-center gap-2">
        {stats && numericValue != null && (
          <>
            <span className="font-mono text-[10px] text-muted-foreground/70">
              p{Math.round(percentile(stats, numericValue))}
            </span>
            <HistogramBrush
              binsAll={stats.binsAll}
              min={stats.min}
              max={stats.max}
              width={96}
              height={18}
              marker={numericValue}
              className="shrink-0 opacity-90"
            />
          </>
        )}
        <span className="truncate font-mono text-foreground">
          {value}
          {unit && <span className="ml-0.5 text-muted-foreground/70">{unit}</span>}
        </span>
      </span>
    </div>
  )
}
