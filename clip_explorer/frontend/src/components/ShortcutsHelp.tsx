import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Kbd } from "./Kbd"

const BINDINGS: Array<[string[], string]> = [
  [["j", "↓"], "next clip"],
  [["k", "↑"], "previous clip"],
  [["space"], "play / pause"],
  [["enter"], "open raw video in a new tab"],
  [["/"], "focus search"],
  [["esc"], "leave search / close"],
  [["shift", "click header"], "add column to sort"],
  [["?"], "this help"],
]

export function ShortcutsHelp({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          {BINDINGS.map(([keys, what]) => (
            <div key={what} className="flex items-center justify-between gap-4 text-sm">
              <span className="text-muted-foreground">{what}</span>
              <span className="flex gap-1">
                {keys.map((k) => (
                  <Kbd key={k}>{k}</Kbd>
                ))}
              </span>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
