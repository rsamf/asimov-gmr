"""Back the clip explorer with a HuggingFace dataset repo instead of local disk.

Retargeted motion derives from AMASS and cannot be redistributed, so the public
Space runs against a PRIVATE dataset repo using the Space's own HF token. The
app itself is open source; the clips are not.

Enable by setting, in the Space's variables/secrets:

    HF_DATASET_REPO=<owner>/<name>     # dataset repo holding a release
    HF_TOKEN=<read token>              # required if that repo is private

Metadata (a few MB of JSON) is downloaded once at startup. Videos are ~1.4 GB
per release, so they are fetched lazily per request and cached on the Space's
local disk — downloading them all at boot would blow the cold-start budget.
"""
import os
import threading

REPO_ID = os.environ.get("HF_DATASET_REPO")
REPO_TYPE = os.environ.get("HF_DATASET_REPO_TYPE", "dataset")
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# JSON/CSV metadata is small; mp4s are not
METADATA_PATTERNS = ["**/*.json", "**/*.csv"]

_lock = threading.Lock()
_root = None


def enabled():
    return bool(REPO_ID)


def bootstrap():
    """Download the release metadata and return a local data root.

    The returned directory has the same shape the app expects locally
    (`asimov/retargeted/tune_*`, `asimov/motions_train/tune_*`, ...), so nothing
    downstream needs to know where it came from.
    """
    global _root
    if _root or not REPO_ID:
        return _root
    with _lock:
        if _root:
            return _root
        from huggingface_hub import snapshot_download
        _root = snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                  token=TOKEN, allow_patterns=METADATA_PATTERNS)
        print(f"[hf] metadata from {REPO_ID} -> {_root}", flush=True)
    return _root


def fetch_video(rel_in_repo):
    """Local path for one video, downloading it on first request.

    `hf_hub_download` caches by content hash, so repeat views are a cache hit
    and a Space restart only re-fetches what is actually watched.
    """
    if not REPO_ID:
        return None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                               filename=rel_in_repo, token=TOKEN)
    except Exception as e:                      # missing file, auth, network
        print(f"[hf] video fetch failed for {rel_in_repo}: {type(e).__name__}: {e}",
              flush=True)
        return None
