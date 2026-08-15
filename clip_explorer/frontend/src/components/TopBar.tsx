import type { Table } from "@tanstack/react-table"
import { ChevronDown, Columns3, Keyboard } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { PRESETS, type Preset } from "@/lib/presets"
import type { ClipRow, DatasetInfo } from "@/types"

interface TopBarProps {
  datasets: DatasetInfo[]
  dataset: string
  onDatasetChange: (name: string) => void
  onApplyPreset: (p: Preset) => void
  table: Table<ClipRow>
  onShowHelp: () => void
}

export function TopBar({
  datasets,
  dataset,
  onDatasetChange,
  onApplyPreset,
  table,
  onShowHelp,
}: TopBarProps) {
  const current = datasets.find((d) => d.name === dataset)
  const hideable = table
    .getAllLeafColumns()
    .filter((c) => c.id !== "clip_name" && c.id !== "has_video")

  return (
    <header className="flex h-11 shrink-0 items-center gap-2 border-b border-border bg-card px-3">
      <div className="flex items-center gap-2.5 pr-3">
        <span className="size-2 rounded-full bg-marker shadow-[0_0_8px_2px_rgba(242,239,231,0.35)]" />
        <h1 className="font-condensed text-[13px] font-semibold uppercase tracking-[0.2em]">
          Asimov <span className="text-muted-foreground">· Clip Explorer</span>
        </h1>
      </div>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" className="font-mono text-xs">
            {current?.label ?? dataset}
            <ChevronDown className="text-muted-foreground" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {datasets.map((d) => (
            <DropdownMenuItem
              key={d.name}
              onSelect={() => onDatasetChange(d.name)}
              className="font-mono text-xs"
            >
              <span className="w-7 text-muted-foreground">{d.name}</span>
              {d.label}
              {d.name === dataset && <span className="ml-auto pl-3">·</span>}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm">
            presets
            <ChevronDown className="text-muted-foreground" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-72">
          {PRESETS.map((p) => (
            <DropdownMenuItem key={p.id} onSelect={() => onApplyPreset(p)}>
              <div className="flex flex-col gap-0.5">
                <span className="text-xs font-medium">{p.label}</span>
                <span className="text-[11px] text-muted-foreground">{p.description}</span>
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <span className="ml-auto" />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" aria-label="Show or hide columns">
            <Columns3 />
            columns
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="max-h-96 overflow-y-auto">
          {hideable.map((c) => (
            <DropdownMenuCheckboxItem
              key={c.id}
              checked={c.getIsVisible()}
              onCheckedChange={(v) => c.toggleVisibility(v === true)}
              onSelect={(e) => e.preventDefault()}
              className="text-xs"
            >
              {c.columnDef.meta?.spec?.t ?? c.id}
            </DropdownMenuCheckboxItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <Button variant="ghost" size="icon-sm" aria-label="Keyboard shortcuts" onClick={onShowHelp}>
        <Keyboard />
      </Button>
    </header>
  )
}
