import { useCallback, useRef, useState } from "react"
import { cn } from "@/lib/utils"

interface HistogramBrushProps {
  /** distribution over all rows — the fixed, muted reference layer */
  binsAll: ArrayLike<number>
  /** distribution over rows passing every OTHER filter — the amber layer */
  binsFiltered?: ArrayLike<number> | null
  min: number
  max: number
  width: number
  height: number
  /** active range selection to draw (and edit, when onRange is set) */
  range?: [number | null, number | null] | null
  /** makes the histogram brushable; called with bin-edge-snapped bounds */
  onRange?: (lo: number | null, hi: number | null) => void
  /** detail-pane mode: draw a marker line at this value instead of a brush */
  marker?: number | null
  className?: string
}

/**
 * The signature element: a small SVG distribution you can brush to set a
 * range filter. Same component renders the detail pane's "where does this
 * clip sit" sparkline via `marker`.
 */
export function HistogramBrush({
  binsAll,
  binsFiltered,
  min,
  max,
  width,
  height,
  range,
  onRange,
  marker,
  className,
}: HistogramBrushProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<[number, number] | null>(null)
  // state drives the preview overlay; the ref is the source of truth so
  // rapid pointer events within one frame don't read a stale closure
  const dragRef = useRef<[number, number] | null>(null)

  const n = binsAll.length
  const span = max - min || 1
  const barW = width / n
  let peak = 1
  for (let i = 0; i < n; i++) peak = Math.max(peak, binsAll[i])
  const barH = (c: number) => (c > 0 ? Math.max(1, (c / peak) * (height - 2)) : 0)

  const xToValue = useCallback(
    (clientX: number) => {
      const rect = svgRef.current!.getBoundingClientRect()
      const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      return min + frac * span
    },
    [min, span],
  )

  // Snap outward to bin edges so a brush never excludes a bar it touches.
  const snap = useCallback(
    (a: number, b: number): [number | null, number | null] => {
      const [va, vb] = a <= b ? [a, b] : [b, a]
      const binW = span / n
      const lo = min + Math.floor((va - min) / binW) * binW
      const hi = min + Math.ceil((vb - min) / binW) * binW
      return [lo <= min ? null : round6(lo), hi >= max ? null : round6(hi)]
    },
    [min, max, span, n],
  )

  const shownRange: [number | null, number | null] | null = drag ? snap(drag[0], drag[1]) : (range ?? null)

  const valueToX = (v: number) => ((v - min) / span) * width

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={cn(onRange && "cursor-crosshair touch-none select-none", className)}
      onPointerDown={
        onRange &&
        ((e) => {
          try {
            e.currentTarget.setPointerCapture(e.pointerId)
          } catch {
            // capture is best-effort — brushing works without it
          }
          const v = xToValue(e.clientX)
          dragRef.current = [v, v]
          setDrag([v, v])
        })
      }
      onPointerMove={
        onRange &&
        ((e) => {
          const d = dragRef.current
          if (!d) return
          dragRef.current = [d[0], xToValue(e.clientX)]
          setDrag(dragRef.current)
        })
      }
      onPointerUp={
        onRange &&
        ((e) => {
          const d = dragRef.current
          if (!d) return
          const [lo, hi] = snap(d[0], xToValue(e.clientX))
          dragRef.current = null
          setDrag(null)
          onRange(lo, hi)
        })
      }
    >
      {Array.from({ length: n }, (_, i) => {
        const hAll = barH(binsAll[i])
        const hSub = binsFiltered ? barH(binsFiltered[i]) : 0
        return (
          <g key={i}>
            {hAll > 0 && (
              <rect
                x={i * barW + 0.5}
                y={height - hAll}
                width={Math.max(0.5, barW - 1)}
                height={hAll}
                fill="var(--muted-foreground)"
                opacity={0.3}
              />
            )}
            {hSub > 0 && (
              <rect
                x={i * barW + 0.5}
                y={height - hSub}
                width={Math.max(0.5, barW - 1)}
                height={hSub}
                fill="var(--rec)"
                opacity={0.85}
              />
            )}
          </g>
        )
      })}
      {shownRange && (shownRange[0] != null || shownRange[1] != null) && (
        <RangeOverlay
          x1={valueToX(shownRange[0] ?? min)}
          x2={valueToX(shownRange[1] ?? max)}
          height={height}
        />
      )}
      {marker != null && (
        <line
          x1={valueToX(Math.min(max, Math.max(min, marker)))}
          x2={valueToX(Math.min(max, Math.max(min, marker)))}
          y1={0}
          y2={height}
          stroke="var(--marker)"
          strokeWidth={1.5}
        />
      )}
    </svg>
  )
}

function RangeOverlay({ x1, x2, height }: { x1: number; x2: number; height: number }) {
  return (
    <g>
      <rect x={x1} y={0} width={Math.max(0, x2 - x1)} height={height} fill="var(--rec)" opacity={0.14} />
      <line x1={x1} x2={x1} y1={0} y2={height} stroke="var(--rec)" strokeWidth={1} />
      <line x1={x2} x2={x2} y1={0} y2={height} stroke="var(--rec)" strokeWidth={1} />
    </g>
  )
}

// friendly bounds for inputs/URLs; 3 decimals is finer than any bin width here
const round6 = (v: number) => Math.round(v * 1e3) / 1e3
