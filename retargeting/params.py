"""Robot registry — purpose-built for asimov.

The asimov MuJoCo XML is NOT vendored here; it comes from
https://github.com/menloresearch/asimov-1 and is located at runtime (see
`resolve_asimov_xml`). The live IK config lives in configs/ at the repo root;
IK_CONFIG_ROOT points there so `--config <name>` resolution keeps working.
"""
import os
import pathlib

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
ASSET_ROOT = REPO / "assets"
IK_CONFIG_ROOT = REPO / "configs"

# path of the XML inside an asimov-1 checkout
_XML_IN_CHECKOUT = pathlib.Path("sim-model") / "xmls" / "asimov.xml"

# Conventional places to look when neither env var is set, so a side-by-side
# clone of asimov-1 just works.
_CANDIDATE_DIRS = (REPO.parent / "asimov-1", REPO / "asimov-1", REPO / "third_party" / "asimov-1")

_HOWTO = """\
The asimov robot description was not found.

Clone it (it is a separate repository and is not redistributed here):

    git clone https://github.com/menloresearch/asimov-1

then point this pipeline at it in any one of these ways:

    export ASIMOV_ROBOT_DIR=/path/to/asimov-1     # the checkout
    export ASIMOV_ROBOT_XML=/path/to/asimov.xml   # or the XML directly
    # or simply clone it next to this repository:  ../asimov-1

Looked in: {looked}"""


def resolve_asimov_xml():
    """Absolute path to asimov.xml, or a RuntimeError explaining how to get it.

    The XML references its meshes relatively (`meshdir=../assets/meshes`), so
    the file must stay inside its checkout — we resolve a path, never copy it.
    """
    looked = []
    env_xml = os.environ.get("ASIMOV_ROBOT_XML")
    if env_xml:
        p = pathlib.Path(env_xml).expanduser()
        if p.is_file():
            return p.resolve()
        looked.append(f"{p} (from ASIMOV_ROBOT_XML)")

    dirs = []
    env_dir = os.environ.get("ASIMOV_ROBOT_DIR")
    if env_dir:
        dirs.append(pathlib.Path(env_dir).expanduser())
    dirs.extend(_CANDIDATE_DIRS)
    for d in dirs:
        p = d / _XML_IN_CHECKOUT
        if p.is_file():
            return p.resolve()
        looked.append(str(p))

    raise RuntimeError(_HOWTO.format(looked="\n  " + "\n  ".join(looked)))


class _LazyPathDict(dict):
    """dict whose callable values resolve on first access and then cache.

    Keeps `ROBOT_XML_DICT["asimov"]` working at every call site while deferring
    the filesystem lookup, so importing this package never requires the robot
    description to be present. Assignment still overrides (used by tests).
    """

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value):
            value = value()
            self[key] = value
        return value


ROBOT_XML_DICT = _LazyPathDict({
    "asimov": resolve_asimov_xml,
})
IK_CONFIG_DICT = {
    "smplx": {
        "asimov": IK_CONFIG_ROOT / "smplx_to_asimov.json",
    },
}
ROBOT_BASE_DICT = {
    "asimov": "pelvis_link",
}
VIEWER_CAM_DISTANCE_DICT = {
    "asimov": 2.5,
}
