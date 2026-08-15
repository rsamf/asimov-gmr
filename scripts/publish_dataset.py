"""Upload a release's metadata and review videos to a PRIVATE HF dataset repo.

This is what the clip explorer Space reads. It uploads only what the UI needs:
the per-clip JSON caches and the review MP4s — never the retargeted pkls or the
training npz.

    publish_dataset.py --root <pipeline out dir> --release <name> --repo <owner>/<name>

The repo is created PRIVATE and must stay private: everything here derives from
AMASS, whose license forbids redistribution. The script refuses to target a
public repo unless you pass --i-have-redistribution-rights.
"""
import argparse
import json
import os
import sys
import tempfile

METADATA = ["manifest.json", "clip_metrics.json", "clip_difficulty.json",
            "fallen_drop.json"]


def stage(root, release, staging, with_videos=True):
    """Assemble the layout the explorer expects, under `staging`."""
    import shutil
    src_ret = os.path.join(root, "retargeted")
    src_train = os.path.join(root, "train")
    src_vid = os.path.join(root, "videos")
    dst_ret = os.path.join(staging, "asimov", "retargeted", f"tune_{release}")
    dst_train = os.path.join(staging, "asimov", "motions_train", f"tune_{release}")
    dst_vid = os.path.join(staging, "asimov", "review_videos", f"tune_{release}")
    os.makedirs(dst_ret, exist_ok=True)
    os.makedirs(dst_train, exist_ok=True)

    for name in METADATA:
        p = os.path.join(src_ret, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_ret, name))
    for name in ("compile_summary.json", "split.json"):
        p = os.path.join(src_train, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst_train, name))

    n_vid = 0
    if with_videos and os.path.isdir(src_vid):
        os.makedirs(dst_vid, exist_ok=True)
        index = []
        for dirpath, _, files in os.walk(src_vid):
            for f in files:
                if not f.endswith(".mp4"):
                    continue
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, src_vid)
                out = os.path.join(dst_vid, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out):
                    os.link(full, out)          # hardlink: no second copy on disk
                index.append(rel)
                n_vid += 1
        # the Space reads this instead of stat-ing every clip over the network
        json.dump(sorted(index), open(os.path.join(dst_vid, "index.json"), "w"))
    return n_vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pipeline --out directory")
    ap.add_argument("--release", required=True, help="release name used in the dataset layout")
    ap.add_argument("--repo", required=True, help="<owner>/<name> dataset repo")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-have-redistribution-rights", action="store_true",
                    help="allow targeting a PUBLIC repo (you must have written "
                         "permission from the AMASS licensor)")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    private = True
    try:
        info = api.repo_info(a.repo, repo_type="dataset")
        private = info.private
    except Exception:
        pass                                    # not created yet -> we create it private
    if not private and not a.i_have_redistribution_rights:
        sys.exit(f"{a.repo} is PUBLIC. Retargeted AMASS motion may not be "
                 f"redistributed; publish to a private repo, or pass "
                 f"--i-have-redistribution-rights if you have written permission.")

    with tempfile.TemporaryDirectory() as staging:
        n = stage(a.root, a.release, staging, with_videos=not a.no_videos)
        total = sum(len(f) for _, _, f in os.walk(staging))
        print(f"staged {total} files ({n} videos) for tune_{a.release}")
        if a.dry_run:
            for dirpath, _, files in os.walk(staging):
                for f in sorted(files)[:3]:
                    print("  ", os.path.relpath(os.path.join(dirpath, f), staging))
            return 0
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=a.repo, repo_type="dataset",
                          folder_path=staging,
                          commit_message=f"publish tune_{a.release}")
    print(f"uploaded -> https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
