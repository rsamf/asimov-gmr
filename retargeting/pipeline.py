"""One command that turns AMASS + the asimov description into a training set.

    asimov-gmr run --amass <AMASS> --robot <asimov-1> --out <OUT>

Stages, in order (each is also a standalone script — this is thin orchestration,
not a reimplementation):

  retarget   AMASS npz  -> <out>/retargeted/**.pkl + manifest.json
  metrics    per-clip float% and pelvis-relative IK residuals
  difficulty easy/medium/hard from five frozen factors
  compile    -> <out>/train/*.npz (rgmt MotionRef format) + compile_summary.json
  fallen     flag references that already trip the training env's fall check
  merge      union the human rejects with the fallen list
  recompile  final compile against the merged list; stamps difficulty + split
  videos     (optional) per-clip review MP4s for the clip explorer

Re-running is safe: `--resume` skips clips that already have a pkl, and the
metrics/difficulty caches are incremental.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATION = os.path.join(REPO, "curation")

STAGES = ["retarget", "metrics", "difficulty", "compile", "fallen", "merge",
          "recompile", "videos"]


def _script(*parts):
    return os.path.join(REPO, *parts)


class Runner:
    def __init__(self, args):
        self.a = args
        self.env = dict(os.environ)
        self.env["ASIMOV_AMASS_DIR"] = args.amass
        if args.robot:
            self.env["ASIMOV_ROBOT_DIR"] = args.robot
        self.retargeted = os.path.join(args.out, "retargeted")
        self.train = os.path.join(args.out, "train")
        self.videos = os.path.join(args.out, "videos")
        self.merged_rejects = os.path.join(args.out, "rejects_plus_fallen.json")
        self.fallen_drop = os.path.join(args.out, "fallen_drop.json")
        self.fallen_report = os.path.join(args.out, "fallen_report.json")

    def run(self, label, argv):
        print(f"\n=== {label} ===\n$ {' '.join(argv)}", flush=True)
        if self.a.dry_run:
            return
        t0 = time.time()
        r = subprocess.run(argv, env=self.env)
        if r.returncode != 0:
            raise SystemExit(f"stage '{label}' failed (exit {r.returncode})")
        print(f"--- {label} done in {time.time() - t0:.0f}s", flush=True)

    # ---- stages -----------------------------------------------------------

    def stage_retarget(self):
        argv = [sys.executable, _script("scripts", "smplx_to_asimov_dataset.py"),
                "--amass", self.a.amass, "--dst", self.retargeted,
                "--workers", str(self.a.workers)]
        if self.a.clips:
            argv += ["--clips_list", self.a.clips]
        else:
            argv += ["--target_hours", str(self.a.target_hours)]
        if self.a.rejects:
            argv += ["--rejects", self.a.rejects]
        if self.a.resume:
            argv += ["--resume"]
        self.run("retarget", argv)

    def stage_metrics(self):
        self.run("metrics", [sys.executable, _script("clip_explorer", "compute_metrics.py"),
                             self.retargeted, "--workers", str(self.a.workers)])

    def stage_difficulty(self):
        self.run("difficulty", [sys.executable,
                                _script("clip_explorer", "compute_difficulty.py"),
                                self.retargeted])

    def _compile(self, label, rejects):
        argv = [sys.executable, _script("scripts", "compile_training_dataset.py"),
                "--dataset", self.retargeted, "--out", self.train,
                "--rejects", rejects]
        if self.a.test_split:
            argv += ["--test_split", self.a.test_split]
        self.run(label, argv)

    def stage_compile(self):
        self._compile("compile", self.a.rejects or os.devnull)

    def stage_fallen(self):
        self.run("fallen", [sys.executable, _script("scripts", "scan_fallen_refs.py"),
                            self.train, "--out", self.fallen_report,
                            "--exclude", self.fallen_drop,
                            "--workers", str(self.a.workers)])

    def stage_merge(self):
        """Union the human reject list with the fallen-reference drop list.

        This step used to be manual — the fallen scan emits a drop list that
        someone had to merge by hand before recompiling, and forgetting it
        silently shipped references that self-terminate in training.
        """
        print(f"\n=== merge ===", flush=True)
        if self.a.dry_run:
            return
        human = json.load(open(self.a.rejects)) if self.a.rejects and \
            os.path.exists(self.a.rejects) else []
        fallen = json.load(open(self.fallen_drop)) if os.path.exists(self.fallen_drop) else []
        merged = sorted(set(human) | set(fallen))
        json.dump(merged, open(self.merged_rejects, "w"), indent=1)
        print(f"{len(human)} rejected + {len(fallen)} fallen -> {len(merged)} "
              f"-> {self.merged_rejects}", flush=True)

    def stage_recompile(self):
        # the compile only ever ADDS exclusions, so the output dir is rebuilt to
        # avoid leaving npz from clips that are now dropped
        if not self.a.dry_run and os.path.isdir(self.train):
            shutil.rmtree(self.train)
        self._compile("recompile", self.merged_rejects)

    def stage_videos(self):
        self.run("videos", [sys.executable, _script("clip_explorer", "render_videos.py"),
                            self.retargeted, self.videos,
                            "--workers", str(self.a.workers)])
        glitch = os.path.join(self.retargeted, "_rejected_glitch")
        if os.path.isdir(glitch) or self.a.dry_run:
            self.run("videos (glitch)", [sys.executable,
                                         _script("clip_explorer", "render_videos.py"),
                                         glitch, self.videos,
                                         "--workers", str(self.a.workers)])


def verify(runner, expected_path):
    """Compare what we produced against a committed expectation."""
    from collections import Counter
    exp = json.load(open(expected_path))
    summary = json.load(open(os.path.join(runner.train, "compile_summary.json")))
    split = json.load(open(os.path.join(runner.train, "split.json")))
    manifest = json.load(open(os.path.join(runner.retargeted, "manifest.json")))
    st = Counter(c["status"] for c in manifest["clips"])
    got = {
        "source_clips": len(manifest["clips"]),
        "retarget.ok": st["ok"],
        "retarget.glitch": st["glitch"],
        "training_set.clips": summary["clips"],
        "training_set.frames": summary["frames"],
        "training_set.hours": summary["hours"],
        "difficulty.easy": sum(e.get("difficulty") == "easy" for e in summary["index"]),
        "difficulty.medium": sum(e.get("difficulty") == "medium" for e in summary["index"]),
        "difficulty.hard": sum(e.get("difficulty") == "hard" for e in summary["index"]),
        "split.train": split["n_train"],
        "split.test": split["n_test"],
    }
    want = {
        "source_clips": exp["source_clips"],
        "retarget.ok": exp["retarget"]["ok"],
        "retarget.glitch": exp["retarget"]["glitch"],
        "training_set.clips": exp["training_set"]["clips"],
        "training_set.frames": exp["training_set"]["frames"],
        "training_set.hours": exp["training_set"]["hours"],
        "difficulty.easy": exp["difficulty"]["easy"],
        "difficulty.medium": exp["difficulty"]["medium"],
        "difficulty.hard": exp["difficulty"]["hard"],
        "split.train": exp["split"]["train"],
        "split.test": exp["split"]["test"],
    }
    print(f"\n=== verify against {os.path.basename(expected_path)} ===")
    bad = 0
    for k, w in want.items():
        g = got[k]
        ok = g == w
        bad += not ok
        print(f"  {'ok  ' if ok else 'DIFF'} {k:24} got {g!r:>10}  expected {w!r}")
    if bad:
        print(f"\n{bad} value(s) differ. Most likely causes: a different AMASS "
              f"snapshot, a modified IK config, or curation lists out of sync.")
    else:
        print("\nexact match — this run reproduces the published release.")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="asimov-gmr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the retargeting pipeline end to end",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("--amass", default=os.environ.get("ASIMOV_AMASS_DIR"),
                   help="AMASS corpus root (<root>/<DATASET>/<subject>/*_poses.npz)")
    r.add_argument("--robot", default=os.environ.get("ASIMOV_ROBOT_DIR"),
                   help="asimov-1 checkout (github.com/menloresearch/asimov-1); "
                        "omit if it sits next to this repo")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    r.add_argument("--clips", default=os.path.join(CURATION, "clips.txt"),
                   help="source clip list to retarget (default: the reference set). "
                        "Pass '' to select a fresh subset with --target_hours instead.")
    r.add_argument("--target_hours", type=float, default=4.5,
                   help="only used when --clips is empty")
    r.add_argument("--rejects", default=os.path.join(CURATION, "rejects.json"),
                   help="human reject list (clip names)")
    r.add_argument("--test_split", default=os.path.join(CURATION, "test_split.json"),
                   help="frozen held-out split")
    r.add_argument("--videos", action="store_true", help="also render review MP4s")
    r.add_argument("--stage", action="append", choices=STAGES,
                   help="run only this stage (repeatable)")
    r.add_argument("--resume", action="store_true",
                   help="skip clips that already have a retargeted pkl")
    r.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the stage commands without running them")
    r.add_argument("--verify", action="store_true",
                   help="compare the result against curation/expected.json")

    a = ap.parse_args(argv)
    if not a.amass:
        ap.error("--amass is required (or set ASIMOV_AMASS_DIR). Obtain AMASS "
                 "(SMPL+H G) from https://amass.is.tue.mpg.de/")
    a.amass = os.path.abspath(os.path.expanduser(a.amass))
    a.out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(a.out, exist_ok=True)

    runner = Runner(a)
    wanted = a.stage or [s for s in STAGES if s != "videos" or a.videos]
    print(f"asimov-gmr: {a.amass} -> {a.out}\nstages: {', '.join(wanted)}")
    for name in STAGES:
        if name in wanted:
            getattr(runner, f"stage_{name}")()

    if a.verify and not a.dry_run:
        expected = os.path.join(CURATION, "expected.json")
        if os.path.exists(expected):
            return 0 if verify(runner, expected) else 1
    print(f"\ndone -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
