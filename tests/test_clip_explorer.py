import csv
import json
import importlib.util
import os
import pathlib

import pytest

from retargeting.utils.clip_names import clip_name_to_rel, read_clip_rows

ROOT = pathlib.Path(__file__).parent.parent


def _load(name, relpath):
    """Import a clip_explorer script by path (the dir is not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


clip_app = _load("clip_explorer_app", "clip_explorer/app.py")
render_videos = _load("render_videos", "clip_explorer/render_videos.py")


def _register(monkeypatch, **overrides):
    """Register a synthetic dataset release for the API tests.

    Releases are discovered from disk, so tests must not assume any particular
    one exists — they register their own pointing at tmp_path.
    """
    d = clip_app.release("test")
    d.update(overrides)
    d.setdefault("train_csv", None)
    d.setdefault("success_csv", None)
    monkeypatch.setitem(clip_app.DATASETS, "test", d)
    return d


HEADER = ["clip_name", "npz_path", "dataset", "passes_succeeded", "steps_survived",
          "survival_frac", "len_frames", "duration_s", "kpe_mm_while_alive",
          "mean_speed_ms", "max_speed_ms", "z_range_m"]
ROWS = [
    ["ACCAD__Female1General_c3d__A7 - crouch", "data/robotics/x.npz", "ACCAD",
     "0", "12", "0.04", "302", "5.03", "63.7", "0.11", "0.343", "0.511"],
    ["CMU__55__55_07", "data/robotics/y.npz", "CMU",
     "0", "87", "1.0", "500", "16.67", "122.4", "0.9", "1.301", "0.359"],
]


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "clips.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(ROWS)
    return p


def test_clip_name_to_rel_nests_and_restores_poses_suffix():
    assert (clip_name_to_rel("ACCAD__Female1General_c3d__A7 - crouch")
            == "ACCAD/Female1General_c3d/A7 - crouch_poses")
    assert clip_name_to_rel("CMU__55__55_07") == "CMU/55/55_07_poses"


def test_clip_name_to_rel_treats_only_first_two_seps_as_separators():
    # every name in the current CSV has exactly two '__', but a clip title is
    # free to contain '__' itself; it must not be split into a subdirectory
    assert clip_name_to_rel("DS__sub__a__b") == "DS/sub/a__b_poses"


def test_clip_name_to_rel_rejects_malformed():
    with pytest.raises(ValueError):
        clip_name_to_rel("ACCAD__missing_clip_field")


def test_read_clip_rows_coerces_numerics_only(csv_file):
    rows = read_clip_rows(csv_file)
    assert len(rows) == 2
    assert rows[0]["len_frames"] == 302              # int
    assert isinstance(rows[0]["len_frames"], int)
    assert rows[0]["survival_frac"] == 0.04          # float
    assert rows[1]["survival_frac"] == 1.0
    assert rows[1]["clip_name"] == "CMU__55__55_07"  # digit-ish text stays text
    assert rows[1]["dataset"] == "CMU"
    assert rows[1]["npz_path"] == "data/robotics/y.npz"


def _make_dataset(tmp_path, rels):
    ds = tmp_path / "ds"
    for rel in rels:
        p = ds / (rel + ".pkl")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return ds


def test_select_pkls_filters_to_csv_clips(csv_file, tmp_path):
    ds = _make_dataset(tmp_path, [
        "ACCAD/Female1General_c3d/A7 - crouch_poses",   # in csv
        "CMU/55/55_07_poses",                           # in csv
        "CMU/55/55_99_poses",                           # not in csv
    ])
    got = render_videos.select_pkls(str(ds), str(csv_file))
    assert sorted(os.path.relpath(p, ds) for p in got) == [
        "ACCAD/Female1General_c3d/A7 - crouch_poses.pkl",
        "CMU/55/55_07_poses.pkl",
    ]


def test_select_pkls_without_csv_returns_whole_dataset(tmp_path):
    ds = _make_dataset(tmp_path, ["CMU/55/55_07_poses", "CMU/55/55_99_poses"])
    assert len(render_videos.select_pkls(str(ds))) == 2


def test_select_pkls_skips_rejected(tmp_path):
    ds = _make_dataset(tmp_path, ["CMU/55/55_07_poses", "CMU/55_rejected/55_99_poses"])
    got = render_videos.select_pkls(str(ds))
    assert [os.path.relpath(p, ds) for p in got] == ["CMU/55/55_07_poses.pkl"]


def test_select_pkls_warns_about_csv_clips_with_no_pkl(csv_file, tmp_path, capsys):
    ds = _make_dataset(tmp_path, ["CMU/55/55_07_poses"])   # ACCAD clip absent
    got = render_videos.select_pkls(str(ds), str(csv_file))
    assert len(got) == 1
    # a silently-dropped clip would read as "fully rendered" later
    assert "no .pkl" in capsys.readouterr().out


@pytest.fixture
def video_dir(tmp_path):
    vd = tmp_path / "videos"
    (vd / "CMU" / "55").mkdir(parents=True)
    (vd / "CMU" / "55" / "55_07_poses.mp4").write_bytes(b"fake mp4 bytes")
    return vd


def test_resolve_video_blocks_escapes(video_dir):
    vd = str(video_dir)
    assert clip_app._resolve_video(vd, "CMU/55/55_07_poses.mp4") is not None
    assert clip_app._resolve_video(vd, "../secret.mp4") is None
    assert clip_app._resolve_video(vd, "CMU/../../secret.mp4") is None
    assert clip_app._resolve_video(vd, "/etc/passwd") is None


def test_resolve_video_allows_symlinked_cache_entries(tmp_path):
    """Regression: a hub-cached release stores each video as a symlink into a
    blob store. Resolving the symlink target puts it outside video_dir, which
    previously 403'd every video after the first fetch."""
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    blob = blob_dir / "deadbeef"
    blob.write_bytes(b"fake mp4 bytes")
    vd = tmp_path / "videos"
    (vd / "CMU" / "55").mkdir(parents=True)
    (vd / "CMU" / "55" / "55_07_poses.mp4").symlink_to(blob)

    got = clip_app._resolve_video(str(vd), "CMU/55/55_07_poses.mp4")
    assert got is not None, "symlinked cache entry must resolve, not 403"
    assert os.path.exists(got)
    # traversal is still refused even though we no longer realpath
    assert clip_app._resolve_video(str(vd), "../blobs/deadbeef") is None


def test_clipname_from_src_parses_dataset_sub_clip():
    name, ds = clip_app._clipname_from_src(
        "/data/amass/CMU/55/55_07_poses.npz")
    assert name == "CMU__55__55_07" and ds == "CMU"


def test_v2_api_joins_manifest_and_metrics_cache(tmp_path, monkeypatch):
    # minimal v2 manifest + metrics cache + a rendered video
    root = tmp_path
    man = {"clips": [
        {"src": "/data/amass/CMU/55/55_07_poses.npz",
         "status": "ok", "overall_sat": 19.9, "ankle_roll_sat": 97.8,
         "peak_vel": 4.9, "glitch": False, "frames": 90, "duration_s": 3.0},
    ]}
    (root / "manifest.json").write_text(json.dumps(man))
    (root / "metrics.json").write_text(json.dumps(
        {"CMU__55__55_07": {"float_pct": 1.5, "pos_err_cm": 2.3,
                            "foot_err_cm": 0.9, "wrist_err_cm": 8.1}}))
    vd = root / "vid"; (vd / "CMU" / "55").mkdir(parents=True)
    (vd / "CMU" / "55" / "55_07_poses.mp4").write_bytes(b"x")
    _register(monkeypatch, manifest=str(root / "manifest.json"), metrics=str(root / "metrics.json"), video_dir=str(vd))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    assert d["total"] == 1
    c = d["clips"][0]
    assert c["clip_name"] == "CMU__55__55_07" and c["dataset"] == "CMU"
    assert c["sat"] == 19.9 and c["ankle"] == 97.8      # from manifest
    assert c["pos_err"] == 2.3 and c["float"] == 1.5     # from metrics cache
    assert c["has_video"] is True
    # default sort is highest saturation first
    assert d["sort"] == {"key": "sat", "dir": -1}


def test_v2_api_empty_when_no_manifest(tmp_path, monkeypatch):
    _register(monkeypatch, manifest=str(tmp_path / "nope.json"))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    assert d["total"] == 0 and d["clips"] == []


def test_api_joins_difficulty_json(tmp_path, monkeypatch):
    root = tmp_path
    man = {"summary": {}, "clips": [
        {"src": "/data/amass/CMU/55/55_07_poses.npz",
         "status": "ok", "overall_sat": 19.9, "ankle_roll_sat": 97.8,
         "peak_vel": 4.9, "glitch": False, "frames": 90, "duration_s": 3.0},
    ]}
    (root / "manifest.json").write_text(json.dumps(man))
    (root / "difficulty.json").write_text(json.dumps(
        {"CMU__55__55_07": {"difficulty": "hard", "level": 2, "driver": "sat",
                            "max_joint_vel": 4.9, "max_root_speed_ms": 0.42,
                            "sat_pct": 19.9, "max_tilt_deg": 12.3}}))
    _register(monkeypatch, manifest=str(root / "manifest.json"), metrics=str(root / "none.json"), difficulty=str(root / "difficulty.json"), video_dir=str(root))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    c = d["clips"][0]
    assert c["difficulty"] == "hard" and c["driver"] == "sat"
    assert c["root_v"] == 0.42 and c["tilt"] == 12.3
    cols = {col["k"] for col in d["columns"]}
    assert {"difficulty", "root_v", "tilt"} <= cols


def test_api_split_column_marks_test_and_train(tmp_path, monkeypatch):
    root = tmp_path
    man = {"summary": {}, "clips": [
        {"src": f"/data/amass/CMU/55/55_{i:02d}_poses.npz",
         "status": "ok", "overall_sat": 1.0, "ankle_roll_sat": 0.0,
         "peak_vel": 1.0, "glitch": False, "frames": 90, "duration_s": 3.0}
        for i in (7, 8)]}
    (root / "manifest.json").write_text(json.dumps(man))
    (root / "summary.json").write_text(json.dumps(
        {"index": [{"clip": "CMU__55__55_07"}, {"clip": "CMU__55__55_08"}]}))
    (root / "test_split.json").write_text(json.dumps(
        {"test": {"easy": ["CMU__55__55_07"], "medium": [], "hard": []}}))
    _register(monkeypatch, manifest=str(root / "manifest.json"), metrics=str(root / "none.json"), difficulty=str(root / "none.json"), train_summary=str(root / "summary.json"), video_dir=str(root))
    monkeypatch.setattr(clip_app, "TEST_SPLIT_PATH", str(root / "test_split.json"))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    by = {c["clip_name"]: c for c in d["clips"]}
    assert by["CMU__55__55_07"]["split"] == "test"
    assert by["CMU__55__55_08"]["split"] == "train"
    assert "split" in {c["k"] for c in d["columns"]}


def test_api_joins_success_csv_pass_column(tmp_path, monkeypatch):
    root = tmp_path
    man = {"summary": {}, "clips": [
        {"src": f"/data/amass/CMU/55/55_{i:02d}_poses.npz",
         "status": "ok", "overall_sat": 1.0, "ankle_roll_sat": 0.0,
         "peak_vel": 1.0, "glitch": False, "frames": 90, "duration_s": 3.0}
        for i in (7, 8)]}
    (root / "manifest.json").write_text(json.dumps(man))
    (root / "success.csv").write_text(
        "clip,split,difficulty,label,success_passes,n_passes\n"
        "CMU__55__55_07,train,easy,failed,1,3\n")
    _register(monkeypatch, manifest=str(root / "manifest.json"), metrics=str(root / "none.json"), difficulty=str(root / "none.json"), success_csv=str(root / "success.csv"), video_dir=str(root))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    by = {c["clip_name"]: c for c in d["clips"]}
    assert by["CMU__55__55_07"]["pass"] == "failed"
    assert by["CMU__55__55_08"]["pass"] is None       # not in the CSV -> em-dash
    assert "pass" in {c["k"] for c in d["columns"]}


def test_api_difficulty_absent_renders_none(tmp_path, monkeypatch):
    """No clip_difficulty.json -> keys present but null (frontend shows em-dash)."""
    root = tmp_path
    man = {"summary": {}, "clips": [
        {"src": "/data/amass/CMU/55/55_07_poses.npz",
         "status": "ok", "overall_sat": 1.0, "ankle_roll_sat": 0.0,
         "peak_vel": 1.0, "glitch": False, "frames": 90, "duration_s": 3.0},
    ]}
    (root / "manifest.json").write_text(json.dumps(man))
    _register(monkeypatch, manifest=str(root / "manifest.json"), metrics=str(root / "none.json"), difficulty=str(root / "none.json"), video_dir=str(root))
    d = clip_app.app.test_client().get("/api/clips?dataset=test").get_json()
    c = d["clips"][0]
    assert c["difficulty"] is None and c["root_v"] is None and c["tilt"] is None


def test_train_metrics_csv_maps_run_fields(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "clip,split,difficulty,in_medium_training,dataset,success_rate,success_passes,"
        "n_passes,mpkpe_global_mm,mpkpe_root_mm,last_pass_kpe_mm,n_frames\n"
        "CMU__55__55_07,train,medium,True,CMU,1.0,3,3,38.4,28.0,38.3,179\n"
        "CMU__55__55_08,train,hard,False,CMU,0.0,0,3,,,,120\n")
    tm = clip_app.load_train_metrics(str(csv_path))
    assert tm["CMU__55__55_07"]["trained"] == "yes"
    assert tm["CMU__55__55_08"]["trained"] == "no"      # hard clip, e+m regime
    assert tm["CMU__55__55_07"]["succ"] == 1.0 and tm["CMU__55__55_07"]["mpkpe_g"] == 38.4
    # a clip that failed every pass reports no mpkpe -> None, not 0
    assert tm["CMU__55__55_08"]["mpkpe_g"] is None


def test_train_metrics_csv_without_the_flag_leaves_trained_unset(tmp_path):
    """Older runs' CSVs have no in_medium_training column -> renders as em-dash."""
    csv_path = tmp_path / "old.csv"
    csv_path.write_text("clip,split,dataset,success_rate,mpkpe_global_mm\n"
                        "CMU__55__55_07,train,CMU,0.67,42.0\n")
    tm = clip_app.load_train_metrics(str(csv_path))
    assert tm["CMU__55__55_07"]["trained"] is None
    assert tm["CMU__55__55_07"]["succ"] == 0.67


def test_index_503_when_dist_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_app, "DIST", str(tmp_path / "nope"))
    r = clip_app.app.test_client().get("/")
    assert r.status_code == 503
    assert b"npm run build" in r.data


def test_index_serves_built_dist(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html>built</html>")
    monkeypatch.setattr(clip_app, "DIST", str(tmp_path))
    r = clip_app.app.test_client().get("/")
    assert r.status_code == 200
    assert b"built" in r.data
    # the shell must never be cached, or stale hashed-asset refs break rebuilds
    assert "no-cache" in r.headers.get("Cache-Control", "") \
        or "max-age=0" in r.headers.get("Cache-Control", "")


def test_assets_served_with_immutable_cache(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("x")
    monkeypatch.setattr(clip_app, "DIST", str(tmp_path))
    r = clip_app.app.test_client().get("/assets/index-abc123.js")
    assert r.status_code == 200
    assert "max-age=31536000" in r.headers.get("Cache-Control", "")



# ---- release caching + video index (Space-facing behaviour) ----

def test_api_caches_until_inputs_change(tmp_path, monkeypatch):
    """A repeat request must not re-parse the manifest, but an edit must show."""
    man = {"summary": {}, "clips": [
        {"src": "/data/amass/CMU/55/55_07_poses.npz", "status": "ok",
         "overall_sat": 1.0, "ankle_roll_sat": 0.0, "peak_vel": 1.0,
         "glitch": False, "frames": 90, "duration_s": 3.0}]}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    _register(monkeypatch, manifest=str(mpath), metrics=str(tmp_path / "none.json"),
              difficulty=str(tmp_path / "none.json"), video_dir=str(tmp_path))
    # identity holds at the builder (the API re-serializes, so compare there)
    first = clip_app.build_manifest_ds("test")
    assert clip_app.build_manifest_ds("test") is first      # served from cache

    man["clips"][0]["overall_sat"] = 42.0
    mpath.write_text(json.dumps(man))
    os.utime(mpath, (0, 0))                    # force a distinct mtime
    fresh = clip_app.build_manifest_ds("test")
    assert fresh is not first and fresh[0]["sat"] == 42.0   # invalidated by the edit
    assert clip_app.app.test_client().get(
        "/api/clips?dataset=test").get_json()["clips"][0]["sat"] == 42.0


def test_video_index_avoids_stat_and_reports_availability(tmp_path, monkeypatch):
    vd = tmp_path / "vid"
    vd.mkdir()
    rel = "CMU/55/55_07_poses.mp4"
    (vd / "index.json").write_text(json.dumps([rel]))
    # index is authoritative: the file itself need not be on local disk (the
    # Space fetches it lazily), and a clip absent from the index reads as absent
    assert clip_app._has_video(str(vd), rel) is True
    assert clip_app._has_video(str(vd), "CMU/55/nope.mp4") is False


def test_has_video_falls_back_to_stat_without_an_index(tmp_path):
    vd = tmp_path / "vid"
    (vd / "CMU" / "55").mkdir(parents=True)
    (vd / "CMU" / "55" / "55_07_poses.mp4").write_bytes(b"x")
    assert clip_app._has_video(str(vd), "CMU/55/55_07_poses.mp4") is True
    assert clip_app._has_video(str(vd), "CMU/55/nope.mp4") is False
