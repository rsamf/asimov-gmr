"""Browse retargeted asimov clips: sortable quality metrics + the retargeted video.

Dataset releases are DISCOVERED under the data root (any
<ROBOTICS_ROOT>/asimov/retargeted/tune_<name>/manifest.json) and switched via
/api/clips?dataset=<name>; each joins its release's manifest, metrics cache,
difficulty file, and any training-run CSV drop-ins. Point ROBOTICS_ROOT at the
output of `asimov-gmr run` to browse your own results.

The UI is a built React app (clip_explorer/frontend/); build it once with
  cd clip_explorer/frontend && npm install && npm run build

Run:  .venv/bin/python clip_explorer/app.py
Open: http://localhost:5001
Read-only — this app writes nothing.
"""
import csv
import glob
import json
import os
import re
import sys

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from retargeting.utils.clip_names import amass_rel, clip_name_to_rel, read_clip_rows

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# app.py is loaded both as a script and by path (tests), so make its own
# directory importable before pulling in the sibling module
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import hf_data  # noqa: E402

# Data root holding the releases to browse, laid out as
#   <root>/asimov/retargeted/tune_<name>/   (manifest + metric caches)
#   <root>/asimov/motions_train/tune_<name>/
#   <root>/asimov/review_videos/tune_<name>/
# Defaults to the pipeline's own output dir, so `asimov-gmr run --out out` then
# `python clip_explorer/app.py` just works. Point ROBOTICS_ROOT elsewhere to
# browse a shared corpus.
ROBO = (hf_data.bootstrap() if hf_data.enabled()
        else os.environ.get("ROBOTICS_ROOT") or os.path.join(REPO, "out"))
DATA_ROOT = ROBO

# Per-clip performance of the trained policy (dropped in by the training run),
# keyed by clip_name; joined into BOTH datasets so success/MPKPE are browsable
# across the whole corpus.
# Each dataset joins the CSV of the training run that consumed it (v2 clips ->
# the v2-era run, v3 -> the v3 run), so succ/mpkpe always reflect the policy
# actually trained on that data.
def _train_csv(env_var, name):
    """Env var wins; else first existing of repo root / _do_not_commit (the
    user archives past runs' CSVs there); else the root path (renders as —)."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    for cand in (os.path.join(REPO, name), os.path.join(REPO, "_do_not_commit", name)):
        if os.path.exists(cand):
            return cand
    return os.path.join(REPO, name)


TRAIN_METRICS_CSV = _train_csv("TRAIN_METRICS_CSV", "clip_metrics.csv")

# canonical held-out test list (release-independent; scripts/make_test_split.py)
TEST_SPLIT_PATH = os.environ.get(
    "TEST_SPLIT_PATH", os.path.join(REPO, "curation", "test_split.json"))
# columns appended to every dataset (concise headers); missing values render as '—'
TRAIN_COLS = [
    ("pass", "pass", "text",
     "Pass/fail verdict from the training run's clip_success CSV drop-in "
     "(its label column): succeeded = the trained policy passed the clip's "
     "evaluation passes, failed = it did not. Sorts failures first. — when "
     "the run provided no success CSV."),
    ("trained", "trained", "text",
     "Was this clip in the training run's own training set (its CSV "
     "in_medium_training flag)? yes = the policy trained on it; no = it was "
     "held out, either as a test clip or by the regime (the easy+medium "
     "regime never trains on hard clips). Read `pass` against this: a failed "
     "'no' clip was never seen during training."),
    ("succ", "succ", "f2",
     "Trained-policy success rate on this clip, from the training run's "
     "clip_metrics.csv drop-in. 1.00 = every rollout tracked the clip to the end. "
     "— until a training CSV is provided."),
    ("mpkpe_g", "mpkpe g", "f1",
     "Trained-policy mean per-keypoint position error (mm) in the GLOBAL frame — "
     "includes base drift. From the training run CSV; — until provided."),
    ("mpkpe_r", "mpkpe r", "f1",
     "Trained-policy mean per-keypoint position error (mm) relative to the ROOT — "
     "pose accuracy with base drift removed. From the training run CSV; — until provided."),
]

# A column: key in the clip dict, header title, a format hint the frontend maps
# to a formatter ("text","int","f1","f2","f3","bool"), and a hover-tooltip
# description shown on the column header.
V2_COLS = [
    ("clip_name", "clip", "text",
     "Source clip identity: AMASS dataset __ subject __ clip. Click a row to "
     "play its retargeted video; arrow keys move the selection."),
    ("dataset", "dataset", "text",
     "AMASS collection the source motion comes from (filter with the chips above)."),
    ("sat", "sat", "f1",
     "Joint-limit saturation: mean % of (frame x joint) cells where a joint sits "
     "within ~2 deg of a position limit — i.e. the average share of asimov's 25 "
     "joints at limit. High values mean the motion demands more range of motion "
     "than asimov has."),
    ("ankle", "ankle", "f1",
     "Ankle-roll saturation: % of frames with an ankle-roll joint at its limit. "
     "Ankle roll is ±20° (0.35 rad) as of v7 — releases up to v6 were built "
     "with a wrong ±5.7° limit, which made this the dominant hotspot there."),
    ("pos_err", "pos err", "f1",
     "Mean IK tracking residual (cm) across all tracked robot bodies: distance "
     "from each body to its human target (the chord-synthesized targets the IK "
     "actually tracked). Measured pelvis-relative, so base placement and "
     "grounding do not inflate it."),
    ("foot", "foot", "f1",
     "Same pelvis-relative residual (cm), feet only. The decisive fidelity "
     "number for locomotion — stance feet are position-pinned, so ~1 cm is "
     "expected on clean clips."),
    ("wrist", "wrist", "f1",
     "Same pelvis-relative residual (cm), wrists only. Arms are position-"
     "tracked with orientation free; unreachable overhead targets stay "
     "unreached rather than lifting the base, so extremes can read high."),
    ("float", "float", "f1",
     "Airborne frames: % of frames where the robot's lowest geometry is more "
     "than 2 cm above the floor. Contact-gated grounding keeps this ~0 except "
     "genuine flight (jumps, running)."),
    ("peak_v", "peak v", "f1",
     "Largest single-frame joint jump in the clip (deg/frame; 40 ≈ 1200 deg/s). "
     "A big value that is also an outlier vs the clip's own motion indicates an "
     "IK discontinuity."),
    ("glitch", "glitch", "bool",
     "Discontinuity flag: peak v > 40 deg/frame AND > 3x the clip's own 95th "
     "percentile — an outlier spike, not sustained fast motion. Flagged clips "
     "are moved to _rejected_glitch/ and excluded from training compile."),
    ("root_v", "root v", "f2",
     "Max root (pelvis) linear speed in m/s, from the retargeted base trajectory "
     "(central difference, so a single grounding step can't spike it). "
     "~1.5 = jog, ~3 = sprint. A difficulty factor."),
    ("root_av", "root av", "f1",
     "Max root (pelvis) angular speed in deg/s (central difference of the base "
     "quaternion). ~210 = brisk turn, ~460 = fast spin/whip. A difficulty factor."),
    ("tilt", "tilt", "f1",
     "Max pelvis tilt from upright (deg) anywhere in the clip. 90 = horizontal "
     "(crawl/cartwheel mid-pose), 180 = fully inverted (handstand). "
     "A difficulty factor."),
    ("difficulty", "diff", "text",
     "easy / medium / hard — worst of five factors, each thresholded at cutoffs "
     "frozen from the v7 corpus (p75/p95, re-frozen after the ankle-roll spec "
     "fix): peak v >= 29.8/46.6 deg per frame, root v >= 1.32/2.54 m/s, "
     "root av >= 211.3/466 deg/s, sat >= 14.2/15.9 %, tilt >= 28.7/48.7 deg. "
     "The factor(s) responsible are in the detail pane (driver)."),
    ("split", "split", "text",
     "Standardized held-out split: test = one of the 60 easy + 60 medium + 60 "
     "hard canonical evaluation clips (frozen, dataset-spread; the easy+medium "
     "training regime evaluates on its 120 easy+medium clips, the full regime "
     "on all 180). train = every other clip in the compiled training set; "
     "blank = not in the training set at all."),
    ("status", "status", "text",
     "ok = clean retarget · glitch = discontinuity flagged (see glitch column) "
     "· error = retargeting failed (source unreadable or IK timeout)."),
    ("removed", "removed", "text",
     "Why this clip is NOT in the compiled training set; blank means it IS included. "
     "glitch = IK discontinuity · rejected = human review · fallen = the reference "
     "itself trips the training env's done (pelvis < 0.12 m, tilted past horizontal, "
     "or head < 0.30 m — cartwheels/handstands) · short = under 1 s · error = retarget "
     "failed. Sort by this column to review exactly what was dropped."),
]

def release(name, root=None):
    """Path bundle for one dataset release (`tune_<name>` under the data root).

    A release is just a directory convention, so releases are discovered rather
    than enumerated — the pipeline's output directory works as-is, and so does a
    dataset snapshot downloaded on a Space.
    """
    root = root or ROBO
    ret = f"{root}/asimov/retargeted/tune_{name}"
    return {
        "label": f"all training ({name})",
        "video_dir": f"{root}/asimov/review_videos/tune_{name}",
        "manifest": f"{ret}/manifest.json",
        "metrics": f"{ret}/clip_metrics.json",
        "difficulty": f"{ret}/clip_difficulty.json",
        "train_summary": f"{root}/asimov/motions_train/tune_{name}/compile_summary.json",
        "human_rejects": f"{root}/asimov/retargeted/tune/rejects.json",
        "fallen_drop": f"{ret}/fallen_drop.json",
        "columns": V2_COLS + TRAIN_COLS,
        "train_csv": _train_csv(f"TRAIN_METRICS_CSV_{name.upper()}",
                                f"clip_metrics_tune_{name}_medium.csv"),
        "success_csv": _train_csv(f"SUCCESS_CSV_{name.upper()}",
                                  f"clip_success_tune_{name}_medium.csv"),
        "sort": ("sat", -1),            # highest joint-limit saturation first
    }


def discover_releases(root=None):
    """{name: release} for every tune_<name>/manifest.json under the data root,
    newest last (natural order, so v10 would sort after v9)."""
    root = root or ROBO
    names = []
    for m in glob.glob(f"{root}/asimov/retargeted/tune_*/manifest.json"):
        name = os.path.basename(os.path.dirname(m))[len("tune_"):]
        if name:
            names.append(name)
    names.sort(key=lambda s: [int(t) if t.isdigit() else t
                              for t in re.split(r"(\d+)", s)])
    return {n: release(n, root) for n in names}


DATASETS = discover_releases()
DEFAULT_DATASET = next(reversed(DATASETS), None)

app = Flask(__name__)


def _dataset(name):
    return DATASETS.get(name) or (DATASETS[DEFAULT_DATASET]
                                 if DEFAULT_DATASET else None)


def _num(v):
    if v in (None, "", "nan", "None"):
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() else f


def load_train_metrics(path=TRAIN_METRICS_CSV):
    """Per-clip trained-policy metrics keyed by clip_name (empty if the CSV is absent)."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["clip"]] = {
            "succ": _num(r.get("success_rate")),
            "mpkpe_g": _num(r.get("mpkpe_global_mm")),
            "mpkpe_r": _num(r.get("mpkpe_root_mm")),
            # what the run itself trained on ("" for older CSVs without the flag)
            "trained": {"True": "yes", "False": "no"}.get(r.get("in_medium_training")),
            # extra fields surface in the detail pane, not as columns
            "succ_passes": _num(r.get("success_passes")), "n_passes": _num(r.get("n_passes")),
            "last_kpe_mm": _num(r.get("last_pass_kpe_mm")), "split": r.get("split"),
        }
    return out


def load_success(path):
    """{clip_name: pass/fail label} from a clip_success CSV (empty if absent)."""
    if not path or not os.path.exists(path):
        return {}
    return {r["clip"]: r["label"] for r in csv.DictReader(open(path)) if r.get("label")}


def _video_index(video_dir):
    """Set of video paths available for a release, or None to stat per clip.

    A release may ship `index.json` (a flat list of relative mp4 paths) next to
    its videos. Reading one file beats ~3.4k stat() calls per request, and it is
    the only workable option when the videos live in a remote repo that is
    fetched lazily.
    """
    idx = os.path.join(video_dir, "index.json")
    try:
        mt = os.path.getmtime(idx)
    except OSError:
        return None
    hit = _VIDEO_INDEX.get(video_dir)
    if not hit or hit[0] != mt:
        _VIDEO_INDEX[video_dir] = (mt, set(json.load(open(idx))))
    return _VIDEO_INDEX[video_dir][1]


_VIDEO_INDEX = {}


def _has_video(video_dir, rel):
    idx = _video_index(video_dir)
    if idx is not None:
        return rel in idx
    return os.path.exists(os.path.join(video_dir, rel))


def _clipname_from_src(src):
    """<amass>/<dataset>/<subdir>/<clip>_poses.npz -> ('<ds>__<sub>__<clip>', '<ds>')."""
    rel = amass_rel(src)
    rel = rel[:-4] if rel.endswith(".npz") else rel        # strip .npz
    ds, sub, clip = rel.split("/", 2)
    if clip.endswith("_poses"):
        clip = clip[:-6]
    return f"{ds}__{sub}__{clip}", ds


def _reject_names(path):
    """rejects-style 'DS/sub/clip_poses.mp4' -> clip_name 'DS__sub__clip'."""
    if not path or not os.path.exists(path):
        return set()
    out = set()
    for r in json.load(open(path)):
        stem = r[:-4] if r.endswith(".mp4") else r
        if stem.endswith("_poses"):
            stem = stem[:-6]
        out.add(stem.replace("/", "__"))
    return out


def _removed_sets(d):
    """Sets used to explain why a clip is absent from the compiled training set.

    compile_summary.json is the authority on what training actually received;
    the reason is then attributed to the filter that would have caught it."""
    summary = d.get("train_summary")
    if not summary or not os.path.exists(summary):
        return None
    return {"kept": {c["clip"] for c in json.load(open(summary)).get("index", [])},
            "human": _reject_names(d.get("human_rejects")),
            "fallen": _reject_names(d.get("fallen_drop"))}


_CACHE = {}


def _cache_key(d):
    """Identity of a release's inputs: paths plus their mtimes.

    Keying on mtime means a rebuilt release (or a test pointing the registry at
    a different directory) invalidates automatically, while repeat requests skip
    re-parsing several MB of JSON — which matters when the data root is a
    network-backed snapshot rather than local disk.
    """
    parts = []
    for k in ("manifest", "metrics", "difficulty", "train_summary",
              "human_rejects", "fallen_drop", "train_csv", "success_csv"):
        p = d.get(k)
        if p:
            try:
                parts.append((p, os.path.getmtime(p)))
            except OSError:
                parts.append((p, None))
    p = TEST_SPLIT_PATH
    parts.append((p, os.path.getmtime(p) if os.path.exists(p) else None))
    return tuple(parts)


def build_manifest_ds(key):
    """Join a release's manifest with its metric caches and training drop-ins."""
    d = DATASETS[key]
    ck = _cache_key(d)
    hit = _CACHE.get(key)
    if hit and hit[0] == ck:
        return hit[1]
    rows = _build_manifest_ds(key, d)
    _CACHE[key] = (ck, rows)
    return rows


def _build_manifest_ds(key, d):
    if not os.path.exists(d["manifest"]):
        return []
    clips = json.load(open(d["manifest"]))["clips"]
    cache = json.load(open(d["metrics"])) if os.path.exists(d["metrics"]) else {}
    diff_path = d.get("difficulty", "")
    diff = json.load(open(diff_path)) if diff_path and os.path.exists(diff_path) else {}
    tsplit = (set().union(*json.load(open(TEST_SPLIT_PATH))["test"].values())
              if os.path.exists(TEST_SPLIT_PATH) else set())
    tm = load_train_metrics(d.get("train_csv") or TRAIN_METRICS_CSV)
    sc = load_success(d.get("success_csv"))
    vdir = d["video_dir"]
    rm = _removed_sets(d)
    out = []
    for c in clips:
        name, ds = _clipname_from_src(c["src"])
        rel = clip_name_to_rel(name) + ".mp4"
        m = cache.get(name, {})
        df = diff.get(name, {})
        row = {
            "clip_name": name, "dataset": ds, "status": c.get("status", "?"),
            "sat": c.get("overall_sat"), "ankle": c.get("ankle_roll_sat"),
            "peak_v": c.get("peak_vel"), "glitch": bool(c.get("glitch")),
            "frames": c.get("frames"), "duration_s": c.get("duration_s"),
            "float": m.get("float_pct"), "pos_err": m.get("pos_err_cm"),
            "foot": m.get("foot_err_cm"), "wrist": m.get("wrist_err_cm"),
            "root_v": df.get("max_root_speed_ms"), "root_av": df.get("max_root_ang_deg_s"),
            "tilt": df.get("max_tilt_deg"),
            "difficulty": df.get("difficulty"), "driver": df.get("driver"),
            "src": c.get("src"), "video": rel, "has_video": _has_video(vdir, rel),
        }
        if rm is not None:
            if name in rm["kept"]:
                row["removed"] = ""
            elif str(c.get("status", "")).startswith("error"):
                row["removed"] = "error"
            elif bool(c.get("glitch")):
                row["removed"] = "glitch"
            elif name in rm["fallen"]:
                row["removed"] = "fallen"
            elif name in rm["human"]:
                row["removed"] = "rejected"
            elif (c.get("duration_s") or 0.0) < 1.0:   # compile's min_seconds gate
                row["removed"] = "short"
            else:
                row["removed"] = "excluded"
        row.update(tm.get(name, {}))
        row["pass"] = sc.get(name)
        # AFTER the CSV join: training CSVs carry their run's own `split`
        # column — the canonical test_split.json is the authority here
        if rm is not None:
            if row.get("removed") == "":
                row["split"] = "test" if name in tsplit else "train"
            else:
                row.pop("split", None)
        out.append(row)
    return out





def _resolve_video(video_dir, name):
    """Absolute path for a requested video, or None if it escapes video_dir."""
    root = os.path.realpath(video_dir)
    full = os.path.realpath(os.path.join(root, name))
    if full != root and not full.startswith(root + os.sep):
        return None
    return full


# Built frontend (clip_explorer/frontend/, Vite): index.html is served
# uncached so rebuilds take effect immediately; /assets files carry a content
# hash in their name, so they are immutable and cached for a year.
DIST = os.path.join(HERE, "frontend", "dist")


@app.route("/")
def index():
    if not os.path.exists(os.path.join(DIST, "index.html")):
        return ("<h1>clip explorer frontend not built</h1>"
                "<p>run: <code>cd clip_explorer/frontend "
                "&amp;&amp; npm install &amp;&amp; npm run build</code></p>", 503)
    return send_from_directory(DIST, "index.html", max_age=0)


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(DIST, "assets"), filename,
                               max_age=31536000)


@app.route("/favicon.svg")
def favicon():
    return send_from_directory(DIST, "favicon.svg")


@app.route("/api/datasets")
def api_datasets():
    return jsonify([{"name": n, "label": d["label"]} for n, d in DATASETS.items()])


@app.route("/api/clips")
def api_clips():
    name = request.args.get("dataset") or DEFAULT_DATASET
    d = _dataset(name)
    if d is None:                       # no releases found under the data root
        return jsonify({"dataset": None, "clips": [], "datasets": [],
                        "columns": [{"k": k, "t": t, "f": f, "d": desc}
                                    for k, t, f, desc in V2_COLS + TRAIN_COLS],
                        "sort": {"key": "sat", "dir": -1},
                        "total": 0, "rendered": 0})
    clips = build_manifest_ds(name) if name in DATASETS else []
    return jsonify({
        "dataset": name,
        "clips": clips,
        "datasets": sorted({c["dataset"] for c in clips}),
        "columns": [{"k": k, "t": t, "f": f, "d": desc} for k, t, f, desc in d["columns"]],
        "sort": {"key": d["sort"][0], "dir": d["sort"][1]},
        "total": len(clips),
        "rendered": sum(c["has_video"] for c in clips),
    })


@app.route("/video/<dataset>/<path:name>")
def video(dataset, name):
    d = _dataset(dataset)
    if d is None:
        abort(404)
    full = _resolve_video(d["video_dir"], name)
    if full is None:                       # escaped the video dir
        abort(403)
    if not os.path.exists(full) and hf_data.enabled():
        # videos are not part of the metadata snapshot; pull this one on demand
        rel = os.path.relpath(full, DATA_ROOT) if full.startswith(DATA_ROOT) else None
        cached = hf_data.fetch_video(rel) if rel else None
        if cached:
            return send_file(cached, mimetype="video/mp4", conditional=True)
    if not os.path.exists(full):
        abort(404)
    return send_file(full, mimetype="video/mp4", conditional=True)  # range support


if __name__ == "__main__":
    if DATASETS:
        print(f"releases: {', '.join(DATASETS)}  (default {DEFAULT_DATASET})")
    else:
        print(f"no releases found under {ROBO}/asimov/retargeted/tune_*\n"
              f"  set ROBOTICS_ROOT, or produce one with: asimov-gmr run --out <dir>")
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5001)), threaded=True)
