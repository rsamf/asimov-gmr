import { useEffect, useRef, useState } from "react"
import { ChevronLeft, ChevronRight, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { videoUrl } from "@/lib/api"
import type { ClipRow } from "@/types"

const RATES = [0.25, 0.5, 1, 1.5, 2]

/**
 * One stable <video> element for the whole session — the src is swapped on
 * selection change instead of recreating the element (which flashes and
 * resets playbackRate).
 */
export function VideoPlayer({ dataset, clip }: { dataset: string; clip: ClipRow | null }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [rate, setRate] = useState(1)

  const src = clip?.has_video ? videoUrl(dataset, clip.video) : null

  useEffect(() => {
    const v = ref.current
    if (!v) return
    if (src) {
      if (v.getAttribute("src") !== src) {
        v.src = src
        v.play().catch(() => {}) // src swap aborts the previous play() — fine
      }
    } else {
      v.removeAttribute("src") // src="" would request the page URL itself
      v.load()
    }
  }, [src])

  // browsers reset playbackRate when a new source loads
  useEffect(() => {
    if (ref.current) ref.current.playbackRate = rate
  }, [rate])

  const fps = clip?.frames && clip.duration_s ? clip.frames / clip.duration_s : 30

  const step = (dir: 1 | -1) => {
    const v = ref.current
    if (!v || !src) return
    v.pause()
    v.currentTime = Math.max(0, v.currentTime + dir / fps)
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg bg-black">
        <video
          id="clip-video"
          ref={ref}
          muted
          autoPlay
          loop
          controls
          playsInline
          className="h-full w-full"
          onLoadedMetadata={(e) => {
            e.currentTarget.playbackRate = rate
          }}
        />
        {!src && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-center text-xs text-muted-foreground">
            {clip ? (
              <>
                <span>no render for this clip yet</span>
                <span className="max-w-full truncate px-4 font-mono">{clip.video}</span>
              </>
            ) : (
              <span>no clip selected</span>
            )}
          </div>
        )}
      </div>
      <div className="flex items-center gap-1">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="font-mono text-xs" disabled={!src}>
              {rate}×
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {RATES.map((r) => (
              <DropdownMenuItem key={r} onSelect={() => setRate(r)} className="font-mono text-xs">
                {r}×{r === rate ? " ·" : ""}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant="ghost" size="icon-sm" disabled={!src} onClick={() => step(-1)} aria-label="Step one frame back" title="Step one frame back">
          <ChevronLeft />
        </Button>
        <Button variant="ghost" size="icon-sm" disabled={!src} onClick={() => step(1)} aria-label="Step one frame forward" title="Step one frame forward">
          <ChevronRight />
        </Button>
        <span className="ml-auto" />
        {src && (
          <Button variant="ghost" size="icon-sm" asChild aria-label="Open raw video" title="Open raw video">
            <a href={src} target="_blank" rel="noreferrer">
              <ExternalLink />
            </a>
          </Button>
        )}
      </div>
    </div>
  )
}
