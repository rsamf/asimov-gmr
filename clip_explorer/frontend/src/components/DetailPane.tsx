import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { columnStats, numVal } from "@/lib/stats"
import { enrichmentOf, GROUP_LABELS, type DetailGroup } from "@/lib/enrichment"
import { formatDetail, formatValue } from "@/lib/format"
import { CopyButton } from "./CopyButton"
import { DIFFICULTY_DOT } from "./ClipTable"
import { MetricRow, MetricSection } from "./MetricSection"
import { VideoPlayer } from "./VideoPlayer"
import type { ClipRow, ClipsResponse, Difficulty } from "@/types"

const GROUP_ORDER: DetailGroup[] = ["ik", "dynamics", "training", "provenance", "other"]

// keys rendered specially, not as generic rows
// (difficulty + driver form the callout badge; n_passes merges into "passes")
const SKIP_KEYS = new Set(["clip_name", "video", "has_video", "driver", "difficulty", "n_passes"])

// Flask alphabetizes JSON keys — restore a meaningful order: server column
// order first, then the detail-only extras.
const EXTRA_ORDER = ["succ_passes", "last_kpe_mm", "frames", "duration_s", "src"]

export function DetailPane({
  clip,
  resp,
  dataset,
}: {
  clip: ClipRow | null
  resp: ClipsResponse
  dataset: string
}) {
  const stats = columnStats(resp)
  const colByKey = new Map(resp.columns.map((c) => [c.k, c]))

  const colOrder = new Map(resp.columns.map((c, i) => [c.k, i]))
  const rank = (k: string) => {
    const ci = colOrder.get(k)
    if (ci != null) return ci
    const ei = EXTRA_ORDER.indexOf(k)
    return 1000 + (ei === -1 ? 99 : ei)
  }

  const groups = new Map<DetailGroup, string[]>()
  if (clip) {
    for (const key of Object.keys(clip)) {
      if (SKIP_KEYS.has(key)) continue
      const g = enrichmentOf(key).group ?? "other"
      const list = groups.get(g) ?? []
      list.push(key)
      groups.set(g, list)
    }
    for (const list of groups.values()) list.sort((a, b) => rank(a) - rank(b))
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
      <VideoPlayer dataset={dataset} clip={clip} />
      {clip && (
        <>
          <div className="flex items-start gap-1">
            <span className="min-w-0 break-all font-mono text-xs leading-5 text-foreground">
              {clip.clip_name}
            </span>
            <span className="ml-auto shrink-0">
              <CopyButton text={clip.clip_name} label="Copy clip name" />
            </span>
          </div>
          <Separator />
          {GROUP_ORDER.map((g) => {
            const keys = groups.get(g)
            if (!keys?.length) return null
            return (
              <MetricSection key={g} title={GROUP_LABELS[g]}>
                {g === "dynamics" && clip.difficulty != null && (
                  <DifficultyCallout difficulty={clip.difficulty} driver={clip.driver ?? null} />
                )}
                {keys.map((key) => {
                  const spec = colByKey.get(key)
                  const e = enrichmentOf(key)
                  const v = clip[key]
                  if (key === "succ_passes") {
                    return (
                      <MetricRow
                        key={key}
                        label="passes"
                        value={
                          v == null ? "—" : `${formatDetail(v)}/${formatDetail(clip.n_passes)}`
                        }
                      />
                    )
                  }
                  const nv = numVal(clip, key)
                  return (
                    <MetricRow
                      key={key}
                      label={spec?.t ?? key.replaceAll("_", " ")}
                      doc={spec?.d}
                      value={spec ? formatValue(spec.f, v) : formatDetail(v)}
                      unit={e.unit}
                      stats={e.histogram ? stats.get(key) : undefined}
                      numericValue={nv}
                    />
                  )
                })}
              </MetricSection>
            )
          })}
        </>
      )}
    </div>
  )
}

function DifficultyCallout({
  difficulty,
  driver,
}: {
  difficulty: Difficulty
  driver: string | null
}) {
  return (
    <Badge variant="outline" className="mb-1 w-fit gap-1.5 font-mono text-[11px]">
      <span className={cn("size-1.5 rounded-full", DIFFICULTY_DOT[difficulty])} />
      {difficulty}
      {driver && <span className="text-muted-foreground">· driver: {driver}</span>}
    </Badge>
  )
}
