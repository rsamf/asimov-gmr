import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { facetCounts } from "@/lib/stats"
import { facetLabel } from "@/lib/enrichment"
import { NULL_TOKEN } from "@/lib/filters"
import type { ClipsResponse, ColumnSpec } from "@/types"

/** Domain-true orderings for the known categoricals; others sort by count. */
const TOKEN_ORDER: Record<string, string[]> = {
  difficulty: ["easy", "medium", "hard", NULL_TOKEN],
  split: ["train", "test", NULL_TOKEN],
  pass: ["succeeded", "failed", NULL_TOKEN],
  trained: ["yes", "no", NULL_TOKEN],
  removed: ["", "glitch", "rejected", "fallen", "short", "error", "excluded", NULL_TOKEN],
  glitch: ["1", "0"],
  has_video: ["1", "0"],
}

interface FacetFilterPanelProps {
  col: ColumnSpec
  resp: ClipsResponse
  /** rows passing every OTHER active filter (null = all rows) */
  exceptMask: Uint8Array | null
  value: string[] | null
  onChange: (tokens: string[] | null) => void
}

export function FacetFilterPanel({ col, resp, exceptMask, value, onChange }: FacetFilterPanelProps) {
  const counts = facetCounts(resp, col.k, exceptMask)
  const allTokens = [...facetCounts(resp, col.k, null).keys()]

  const order = TOKEN_ORDER[col.k]
  const tokens = order
    ? [...order.filter((t) => allTokens.includes(t)), ...allTokens.filter((t) => !order.includes(t))]
    : allTokens.sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))

  const toggle = (token: string, on: boolean) => {
    const next = on ? [...(value ?? []), token] : (value ?? []).filter((t) => t !== token)
    onChange(next.length ? next : null)
  }

  return (
    <div className="flex w-56 flex-col gap-1">
      <PanelTitle title={col.t} onClear={value ? () => onChange(null) : undefined} />
      <div className="flex max-h-72 flex-col gap-0.5 overflow-y-auto">
        {tokens.map((token) => {
          const checked = value?.includes(token) ?? false
          return (
            <label
              key={token}
              className="flex cursor-pointer items-center gap-2 rounded-sm px-1.5 py-1 text-xs hover:bg-accent"
            >
              <Checkbox checked={checked} onCheckedChange={(c) => toggle(token, c === true)} />
              <span className="min-w-0 flex-1 truncate">{facetLabel(col.k, token)}</span>
              <span className="font-mono text-[11px] text-muted-foreground">
                {(counts.get(token) ?? 0).toLocaleString()}
              </span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

export function PanelTitle({ title, onClear }: { title: string; onClear?: () => void }) {
  return (
    <div className="mb-1 flex items-center justify-between">
      <span className="font-condensed text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {title}
      </span>
      {onClear && (
        <Button variant="ghost" size="xs" onClick={onClear}>
          clear
        </Button>
      )}
    </div>
  )
}
