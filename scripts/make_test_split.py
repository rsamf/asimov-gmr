"""Select the canonical, difficulty-balanced held-out TEST split.

One-time (then FROZEN, like the difficulty cutoffs): picks N clips per
difficulty bucket from a compiled release's kept set, round-robin across AMASS
datasets inside each bucket so no collection dominates, deterministic under
--seed. Writes curation/test_split.json —
the release-independent standard the compile step derives per-release
split.json files from (train regime easy+medium evaluates on the easy+medium
buckets; the full regime on all three). Refuses to overwrite without --force;
future releases that drop a test clip leave a reported hole, never a top-up.

  make_test_split.py [--compiled <motions_train_dir>] [--per_bucket 60] [--seed 0]
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CANONICAL = os.path.join(REPO, "curation", "test_split.json")
PENDING = os.path.join(REPO, "_do_not_commit", "new_rejections.txt")
LABELS = ("easy", "medium", "hard")


def select(index, per_bucket, seed, pending=()):
    """{label: [stems]} — per bucket, round-robin one clip per AMASS dataset
    (seeded shuffle within each dataset) until per_bucket picks."""
    pend = set(pending)
    buckets = defaultdict(lambda: defaultdict(list))   # label -> dataset -> stems
    for e in index:
        stem = e["clip"]
        if stem in pend or e.get("difficulty") not in LABELS:
            continue
        buckets[e["difficulty"]][stem.split("__", 1)[0]].append(stem)
    out = {}
    for label in LABELS:
        rng = random.Random(f"{seed}:{label}")         # str-seeded: stable across runs
        queues = {}
        for ds in sorted(buckets[label]):
            lst = sorted(buckets[label][ds])
            rng.shuffle(lst)
            queues[ds] = lst
        picked = []
        while len(picked) < per_bucket and any(queues.values()):
            for ds in sorted(queues):
                if len(picked) >= per_bucket:
                    break
                if queues[ds]:
                    picked.append(queues[ds].pop(0))
        out[label] = sorted(picked)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", required=True,
                    help="compiled release dir (compile_summary.json = the kept pool + labels)")
    ap.add_argument("--out", default=CANONICAL)
    ap.add_argument("--per_bucket", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pending", default=PENDING,
                    help="optional drop-in file of pending rejection stems to exclude")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing canonical file (it is FROZEN otherwise)")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"{a.out} exists — the canonical split is frozen; use --force to redo")
    index = json.load(open(os.path.join(a.compiled, "compile_summary.json")))["index"]
    pending = set()
    if a.pending and os.path.exists(a.pending):
        pending = {ln.strip() for ln in open(a.pending) if ln.strip()}
    test = select(index, a.per_bucket, a.seed, pending)

    for label in LABELS:
        spread = Counter(s.split("__", 1)[0] for s in test[label])
        print(f"{label:>6} ({len(test[label])}): " +
              "  ".join(f"{d} {n}" for d, n in sorted(spread.items())))
    doc = {"created": time.strftime("%Y-%m-%d"), "seed": a.seed,
           "source_release": os.path.basename(os.path.normpath(a.compiled)),
           "per_bucket": a.per_bucket,
           "sampling": "round-robin over AMASS datasets within each difficulty "
                       "bucket, seeded shuffle per dataset; FROZEN — releases that "
                       "drop a test clip leave a reported hole, never a top-up",
           "test": test}
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"WROTE {a.out}  ({sum(len(v) for v in test.values())} test clips)")


if __name__ == "__main__":
    main()
