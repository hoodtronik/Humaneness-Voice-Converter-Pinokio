"""Command-line conversion with the same engine and presets as the GUI (no server needed).

  env/Scripts/python cli.py --source take.wav --preset "Finding Eli/Eli" --out eli_take.wav
  env/Scripts/python cli.py --source take.mp4 --target voice_8s.wav --out out.wav --strength 1.0 --gender "Boost female"
"""
import argparse
import json
import random
import sys

import soundfile as sf

import presets
from engine import Engine, GENDER_TARGET, decode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="performance: audio or video, 1 s to 5 min")
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--target", help="voice reference clip, 5-10 s, one speaker")
    ref.add_argument("--preset", help="'Group/Name' or a preset name from the voice library")
    ap.add_argument("--out", required=True, help="output WAV (mono 24 kHz)")
    ap.add_argument("--seed", type=int, help="default: random (printed in the report)")
    ap.add_argument("--strength", type=float, help="conditioning strength (default 1.0, or the preset's saved value)")
    ap.add_argument("--gender", choices=["Off", *GENDER_TARGET], help="experimental gender boost (default Off)")
    ap.add_argument("--candidates", type=int, choices=(1, 4, 8, 16), default=1)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--metrics", help="optional JSON report path")
    ap.add_argument("--list-presets", action="store_true", help="print the voice library and exit")
    if "--list-presets" in sys.argv:
        for p in presets.all_presets():
            print(f"{p['group']}/{p['name']}  ({p['gender']})")
        return
    a = ap.parse_args()
    settings = {}
    target = a.target
    if a.preset:
        p = presets.find(a.preset)
        if not p:
            sys.exit(f"No preset named {a.preset!r}; see --list-presets")
        target, settings = p["file"], p.get("settings") or {}
    strength = a.strength if a.strength is not None else settings.get("strength", 1.0)
    gender = a.gender or settings.get("gender_boost", "Off")
    seed = a.seed if a.seed is not None else random.randint(0, 2**32 - 1)
    engine = Engine()
    engine.load()
    for wave, status, report in engine.convert(decode(a.source), decode(target), seed, a.flow_steps, strength,
                                               gender, a.candidates, a.precision):
        print(status, flush=True)
    sf.write(a.out, wave, 24000, subtype="PCM_16")
    if a.metrics:
        with open(a.metrics, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({"output": a.out, "seed": seed, "strength": strength, "gender_boost": gender,
                      "seconds": report["seconds"], "sections": len(report["sections"])}))


if __name__ == "__main__":
    main()
