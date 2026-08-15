import { useEffect } from "react"

export interface KeyboardHandlers {
  onMove: (delta: number) => void
  onTogglePlay: () => void
  onOpen: () => void
  onFocusSearch: () => void
  onHelp: () => void
}

export function useKeyboard(handlers: KeyboardHandlers) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) {
        return
      }
      switch (e.key) {
        case "ArrowDown":
        case "j":
          e.preventDefault()
          handlers.onMove(1)
          break
        case "ArrowUp":
        case "k":
          e.preventDefault()
          handlers.onMove(-1)
          break
        case " ":
          e.preventDefault()
          handlers.onTogglePlay()
          break
        case "Enter":
        case "o":
          handlers.onOpen()
          break
        case "/":
          e.preventDefault()
          handlers.onFocusSearch()
          break
        case "?":
          handlers.onHelp()
          break
      }
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [handlers])
}
