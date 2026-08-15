import type { ColumnFiltersState, SortingState } from "@tanstack/react-table"

/**
 * Built-in views encoding the real triage workflows. A preset is nothing but
 * a filter+sort application — pure client state, nothing persisted.
 */
export interface Preset {
  id: string
  label: string
  description: string
  filters: ColumnFiltersState
  sorting: SortingState
}

export const PRESETS: Preset[] = [
  {
    id: "training-failures",
    label: "Training failures",
    description: "Clips the trained policy failed, worst success rate first",
    filters: [{ id: "pass", value: { kind: "facet", values: ["failed"] } }],
    sorting: [{ id: "succ", desc: false }],
  },
  {
    id: "glitches",
    label: "Glitches",
    description: "IK discontinuities excluded from training, biggest spike first",
    filters: [{ id: "removed", value: { kind: "facet", values: ["glitch"] } }],
    sorting: [{ id: "peak_v", desc: true }],
  },
  {
    id: "test-split",
    label: "Test split",
    description: "The frozen 60+60+60 held-out evaluation clips",
    filters: [{ id: "split", value: { kind: "facet", values: ["test"] } }],
    sorting: [{ id: "difficulty", desc: true }],
  },
  {
    id: "worst-saturation",
    label: "Worst saturation",
    description: "Highest joint-limit saturation first, nothing filtered",
    filters: [],
    sorting: [{ id: "sat", desc: true }],
  },
  {
    id: "removed-review",
    label: "Removed from training",
    description: "Everything the compile dropped, grouped by reason",
    filters: [
      {
        id: "removed",
        value: {
          kind: "facet",
          values: ["glitch", "rejected", "fallen", "short", "error", "excluded"],
        },
      },
    ],
    sorting: [{ id: "removed", desc: false }],
  },
]
