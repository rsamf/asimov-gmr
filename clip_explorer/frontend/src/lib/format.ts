import type { FormatHint } from "@/types"

export const MISSING = "—"

/** Render a raw cell value per the server's format hint. */
export function formatValue(f: FormatHint, v: unknown): string {
  if (f === "bool") return v ? "✓" : "·"
  if (v == null) return MISSING
  switch (f) {
    case "int":
      return String(v)
    case "f1":
      return Number(v).toFixed(1)
    case "f2":
      return Number(v).toFixed(2)
    case "f3":
      return Number(v).toFixed(3)
    default:
      return String(v)
  }
}

/** Compact display for arbitrary detail-pane values (unknown keys included). */
export function formatDetail(v: unknown): string {
  if (v == null || v === "") return MISSING
  if (typeof v === "boolean") return v ? "yes" : "no"
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2)
  return String(v)
}
