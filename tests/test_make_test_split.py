import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import make_test_split as ms  # noqa: E402


def _index(spec):
    """spec: {label: {dataset: n_clips}} -> compile_summary-style index list."""
    out = []
    for label, dss in spec.items():
        for ds, n in dss.items():
            for i in range(n):
                out.append({"clip": f"{ds}__sub__{label}{i:02d}", "difficulty": label})
    return out


def test_exact_counts_and_determinism():
    idx = _index({"easy": {"A": 10, "B": 10}, "medium": {"A": 10}, "hard": {"C": 10}})
    a = ms.select(idx, per_bucket=4, seed=0)
    b = ms.select(idx, per_bucket=4, seed=0)
    assert a == b                                    # byte-stable across runs
    assert {k: len(v) for k, v in a.items()} == {"easy": 4, "medium": 4, "hard": 4}
    assert ms.select(idx, per_bucket=4, seed=1) != a  # seed actually matters


def test_round_robin_spreads_across_datasets():
    idx = _index({"easy": {"A": 10, "B": 10, "C": 10}})
    got = ms.select(idx, per_bucket=6, seed=0)["easy"]
    per_ds = {d: sum(s.startswith(d + "__") for s in got) for d in "ABC"}
    assert per_ds == {"A": 2, "B": 2, "C": 2}


def test_small_dataset_exhausts_then_round_robin_continues():
    idx = _index({"easy": {"A": 1, "B": 10}})
    got = ms.select(idx, per_bucket=4, seed=0)["easy"]
    per_ds = {d: sum(s.startswith(d + "__") for s in got) for d in "AB"}
    assert per_ds == {"A": 1, "B": 3}


def test_bucket_smaller_than_quota_takes_everything():
    idx = _index({"easy": {"A": 2}, "medium": {"A": 5}, "hard": {"A": 5}})
    got = ms.select(idx, per_bucket=4, seed=0)
    assert len(got["easy"]) == 2                     # pool exhausted, no crash


def test_pending_rejections_are_excluded():
    idx = _index({"easy": {"A": 3}})
    names = sorted(e["clip"] for e in idx)
    got = ms.select(idx, per_bucket=3, seed=0, pending={names[0]})
    assert names[0] not in got["easy"] and len(got["easy"]) == 2


def test_entries_without_difficulty_are_ignored():
    idx = _index({"easy": {"A": 2}}) + [{"clip": "A__sub__nolabel"}]
    got = ms.select(idx, per_bucket=5, seed=0)
    assert all("nolabel" not in s for s in got["easy"])


def test_all_picks_come_from_pool():
    idx = _index({"easy": {"A": 5, "B": 5}, "medium": {"C": 5}, "hard": {"D": 5}})
    pool = {e["clip"] for e in idx}
    got = ms.select(idx, per_bucket=3, seed=0)
    assert set().union(*got.values()) <= pool


# ---- compile-side derivation (split stamped into summary + rgmt split.json) ----

import json  # noqa: E402

import compile_training_dataset as ctd  # noqa: E402


def _canonical(tmp_path, test):
    p = tmp_path / "test_split.json"
    p.write_text(json.dumps({"seed": 0, "per_bucket": 2, "test": test}))
    return str(p)


def test_apply_split_stamps_index_and_writes_rgmt_json(tmp_path):
    canon = _canonical(tmp_path, {"easy": ["A__s__e0"], "medium": ["B__s__m0"], "hard": []})
    index = [{"clip": "A__s__e0"}, {"clip": "B__s__m0"}, {"clip": "C__s__x0"}]
    test_set = ctd.load_test_split(canon)
    ctd.apply_split(index, test_set, str(tmp_path), canon)
    assert [e["split"] for e in index] == ["test", "test", "train"]
    sj = json.loads((tmp_path / "split.json").read_text())
    assert sj["train"] == ["C__s__x0"] and sorted(sj["test"]) == ["A__s__e0", "B__s__m0"]
    assert sj["n_train"] == 1 and sj["n_test"] == 2
    assert sj["missing_test_clips"] == []
    assert set(sj["train"]).isdisjoint(sj["test"])


def test_apply_split_reports_test_clips_this_release_dropped(tmp_path, capsys):
    canon = _canonical(tmp_path, {"easy": ["A__s__e0", "GONE__s__e1"], "medium": [], "hard": []})
    index = [{"clip": "A__s__e0"}]
    ctd.apply_split(index, ctd.load_test_split(canon), str(tmp_path), canon)
    sj = json.loads((tmp_path / "split.json").read_text())
    assert sj["missing_test_clips"] == ["GONE__s__e1"]   # hole, not a top-up
    assert "GONE__s__e1" not in sj["test"]
    assert "WARNING" in capsys.readouterr().out


def test_load_test_split_missing_file_is_none(tmp_path):
    assert ctd.load_test_split(str(tmp_path / "nope.json")) is None
