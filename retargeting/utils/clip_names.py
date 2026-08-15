"""Map compiled clip names back onto the retargeted files on disk.

The training CSVs (e.g. ``untrackable_clips_n7.csv``) name a clip
``<dataset>__<subdir>__<clip>`` with AMASS's ``_poses`` suffix stripped, while
the retargeted datasets keep both the suffix and the folder nesting.
"""
import csv

#: Columns that must never be parsed as numbers — clip names like "55_07" and
#: dataset names would otherwise be mangled into ints.
TEXT_COLS = frozenset({"clip_name", "npz_path", "dataset"})


def clip_name_to_rel(clip_name):
    """``'ACCAD__F1_c3d__A7 - crouch'`` -> ``'ACCAD/F1_c3d/A7 - crouch_poses'``.

    Returns a relative path with no extension; callers append ``.pkl``/``.mp4``.
    Only the first two ``__`` are separators, so a clip title may contain ``__``.
    """
    parts = clip_name.split("__", 2)
    if len(parts) != 3:
        raise ValueError(
            f"expected '<dataset>__<subdir>__<clip>', got {clip_name!r}")
    dataset, subdir, clip = parts
    return f"{dataset}/{subdir}/{clip}_poses"


def amass_rel(src, amass_root=None):
    """``<amass>/CMU/55/55_07_poses.npz`` -> ``'CMU/55/55_07_poses.npz'``.

    AMASS always nests ``<root>/<dataset>/<subject>/<clip>.npz``, so the last
    three path segments identify a clip regardless of where the corpus lives.
    Manifests store absolute source paths from whichever machine produced them,
    which is why this is resolved from the path shape rather than from a root
    that may no longer exist (or may never have been named ``motions/``).
    """
    p = str(src).replace("\\", "/")
    if amass_root:
        root = str(amass_root).replace("\\", "/").rstrip("/") + "/"
        if p.startswith(root):
            return p[len(root):]
    parts = [seg for seg in p.split("/") if seg]
    if len(parts) < 3:
        raise ValueError(f"cannot derive an AMASS-relative path from {src!r}")
    return "/".join(parts[-3:])


def amass_clip_name(src, amass_root=None):
    """``<amass>/CMU/55/55_07_poses.npz`` -> ``('CMU__55__55_07', 'CMU')``."""
    rel = amass_rel(src, amass_root)
    dataset, subdir, clip = rel.split("/", 2)
    clip = clip[:-4] if clip.endswith(".npz") else clip
    if clip.endswith("_poses"):
        clip = clip[:-6]
    return f"{dataset}__{subdir}__{clip}", dataset


def _coerce(key, value):
    if key in TEXT_COLS or value is None:
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_clip_rows(csv_path):
    """Parse a clip-metrics CSV, coercing numeric columns to int/float."""
    with open(csv_path, newline="") as f:
        return [{k: _coerce(k, v) for k, v in row.items()}
                for row in csv.DictReader(f)]
