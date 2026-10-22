"""Voice presets: the built-in public library plus your own presets, organised in groups.

Built-in voices come from voices/library.json (read-only). Your presets live in
voices/custom/presets.json with their audio under voices/custom/<group>/, which is
git-ignored so private voices (e.g. a film's cast) never reach the public repo.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from engine import REF_MAX_S, decode

ROOT = Path(__file__).resolve().parent
VOICES = ROOT / "voices"
CUSTOM = VOICES / "custom"
STORE = CUSTOM / "presets.json"
BUILTIN_GROUP = "Starter voices (VCTK)"
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"}


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "voice"


def _read():
    if STORE.exists():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {"groups": [], "presets": []}


def _write(data):
    CUSTOM.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def builtin():
    data = json.loads((VOICES / "library.json").read_text(encoding="utf-8"))
    return [{"id": v["id"], "name": v["name"], "group": BUILTIN_GROUP, "gender": v["gender"],
             "notes": f"{v['accent']} · {v['credit']}", "file": str(VOICES / v["file"]), "settings": {},
             "builtin": True} for v in data["voices"]]


def all_presets():
    custom = [{**p, "file": str(CUSTOM / p["file"]), "builtin": False} for p in _read()["presets"]]
    return builtin() + custom


def groups():
    data = _read()
    names = list(dict.fromkeys(data["groups"] + [p["group"] for p in data["presets"]]))
    return [BUILTIN_GROUP] + names


def get(preset_id):
    return next((p for p in all_presets() if p["id"] == preset_id), None)


def find(label):
    """Look up 'Group/Name' or a bare name (first match), for the headless driver."""
    group, _, name = label.rpartition("/")
    for p in all_presets():
        if p["name"].lower() == name.lower() and (not group or p["group"].lower() == group.lower()):
            return p
    return None


def add_group(name):
    name = (name or "").strip()
    if not name or name == BUILTIN_GROUP:
        raise ValueError("Enter a new group name.")
    data = _read()
    if name not in data["groups"]:
        data["groups"].append(name)
        _write(data)


def create(name, group, gender, notes, audio_path, settings=None):
    name, group = (name or "").strip(), (group or "").strip()
    if not name or not group or group == BUILTIN_GROUP:
        raise ValueError("A preset needs a name and one of your own groups.")
    if not audio_path:
        raise ValueError("Add a reference clip first.")
    wave = decode(audio_path)[:int(REF_MAX_S * 24000)]
    if len(wave) < 3 * 24000:
        raise ValueError("The reference clip must be at least 3 s (5-10 s works best).")
    data = _read()
    if any(p["name"].lower() == name.lower() and p["group"] == group for p in data["presets"]):
        raise ValueError(f"'{name}' already exists in {group}.")
    rel = Path(slug(group)) / f"{slug(name)}.flac"
    (CUSTOM / rel).parent.mkdir(parents=True, exist_ok=True)
    sf.write(CUSTOM / rel, wave / max(1e-6, float(np.abs(wave).max())) * 0.89, 24000, subtype="PCM_16")
    preset = {"id": f"custom-{slug(group)}-{slug(name)}-{int(time.time())}", "name": name, "group": group,
              "gender": gender or "Unspecified", "notes": notes or "", "file": rel.as_posix(),
              "settings": settings or {}, "created": time.strftime("%Y-%m-%d %H:%M")}
    data["presets"].append(preset)
    if group not in data["groups"]:
        data["groups"].append(group)
    _write(data)
    return preset


def update(preset_id, **fields):
    data = _read()
    for p in data["presets"]:
        if p["id"] == preset_id:
            p.update({k: v for k, v in fields.items() if v is not None})
            if p["group"] not in data["groups"]:
                data["groups"].append(p["group"])
            _write(data)
            return p
    raise ValueError("Built-in voices can't be edited; save a copy as your own preset instead.")


def delete(preset_id):
    data = _read()
    keep = [p for p in data["presets"] if p["id"] != preset_id]
    if len(keep) == len(data["presets"]):
        raise ValueError("Built-in voices can't be deleted.")
    gone = next(p for p in data["presets"] if p["id"] == preset_id)
    (CUSTOM / gone["file"]).unlink(missing_ok=True)
    data["presets"] = keep
    _write(data)


def import_folder(folder, group, gender):
    """Every audio/video file in `folder` becomes a preset (named after the file) in `group`."""
    folder = Path((folder or "").strip().strip('"'))
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    made, skipped = [], []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in AUDIO_EXT:
            continue
        try:
            made.append(create(f.stem, group, gender, f"Imported from {f.name}", str(f)))
        except ValueError as exc:
            skipped.append(f"{f.name}: {exc}")
    return made, skipped
