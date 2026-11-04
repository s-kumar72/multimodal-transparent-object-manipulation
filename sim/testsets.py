"""
Test sets defined in sim/configs/manifest.json: one scene config, split and seed list per (level, test set).

  cfg, seeds, objects = test_set("L3", "g_heldout_all")
"""
import json
from pathlib import Path

from config import _load_yaml, _merge, make_config
from env import OBJECT_NAMES

MANIFEST = Path(__file__).parent / "configs" / "manifest.json"

def manifest():
    return json.loads(MANIFEST.read_text())

def test_set(level, name, n=None, seed_start=None):
    """Returns (SceneConfig, seeds, objects); episode k uses seeds[k] and objects[seeds[k] % len(objects)]."""
    m = manifest()
    t = m["test_sets"][name]
    d = _merge(_load_yaml(m["levels"][level]), t.get("overrides", {}))
    cfg = make_config(d, split=t["split"], heldout=t["heldout"])
    start = m["seed_start"] if seed_start is None else seed_start
    return cfg, list(range(start, start + (n or m["n"]))), t.get("objects", OBJECT_NAMES)
