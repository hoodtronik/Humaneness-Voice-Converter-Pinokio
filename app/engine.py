"""Conversion engine: LAION's tested infer.py building blocks, with the models kept loaded,
long performances split into overlapping sections, and the two knobs the GUI exposes
(conditioning strength, experimental gender boost).
"""
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "humaneness-vc"
SECTION_S, OVERLAP_S, MIN_TAIL_S = 20.0, 2.0, 4.0
MAX_SOURCE_S, REF_MAX_S, REF_MIN_S = 300.0, 10.0, 3.0
# CLAUDE-NOTE: GEND direction measured 2026-10-22 with Ears Medium on VCTK: female 0.7-2.5, male 4.7-4.9
# (0-6 scale). The boost overrides the target's GEND conditioning score; LAION ships no boost, so it is
# experimental and Off by default.
GENDER_TARGET = {"Boost male": 5.5, "Boost female": 0.5}


def _load_infer():
    spec = importlib.util.spec_from_file_location("humaneness_infer", MODEL_DIR / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(path, rate=24000):
    """Any audio or video file -> mono float32 at `rate`, via the bundled ffmpeg."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(rate), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise ValueError(f"Could not read audio from {Path(path).name}: is there a sound track?")
    return np.frombuffer(out.stdout, np.float32).copy()


class Engine:
    def __init__(self):
        self.infer = None

    def load(self):
        if self.infer is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required (LAION's validated inference path).")
        torch.set_num_threads(4)
        self.device = torch.device("cuda:0")
        self.infer = _load_infer()
        self.model = self.infer.load_converter(MODEL_DIR, self.device)
        self.ears, self.extractor, self.norm = self.infer.load_ears(MODEL_DIR / "conditioning/ears-medium", self.device)
        self.names = json.loads((MODEL_DIR / "conditioning/score_order.json").read_text())["score_order"]
        self.calibration = json.loads((MODEL_DIR / "conditioning/reward_calibration.json").read_text())

    def _scale(self, raw):
        s = raw[:99].copy()
        s[:98] /= 6.0
        s[98] /= 10.0
        return s

    def sections(self, n, rate):
        size, hop = int(SECTION_S * rate), int((SECTION_S - OVERLAP_S) * rate)
        if n <= size:
            return [(0, n)]
        starts = list(range(0, n - size + hop, hop))
        # A very short last section converts poorly; let the previous one run to the end instead (<= 24 s).
        if n - starts[-1] < MIN_TAIL_S * rate and len(starts) > 1:
            starts.pop()
        return [(s, min(n, s + size) if i < len(starts) - 1 else n) for i, s in enumerate(starts)]

    def _rank(self, waves, source_raw, reference):
        infer, norm, terms = self.infer, self.norm, self.calibration["terms"]
        top2 = np.argsort(source_raw[:40])[-2:]
        ce_i, pq_i = norm["score_names"].index("audiobox:CE"), norm["score_names"].index("audiobox:PQ")
        out_raw = np.concatenate([infer.ears_raw_scores(waves[i:i + 8], self.ears, self.extractor, norm, self.device)
                                  for i in range(0, len(waves), 8)], axis=0)
        target = reference["embedding"].float().cpu().numpy().reshape(-1)
        target /= max(float(np.linalg.norm(target)), 1e-8)
        w16 = np.stack([infer.resample(w, 24000, 16000) for w in waves])
        with torch.inference_mode():
            camp = self.model.speaker_encoder.inference(torch.from_numpy(w16).to(self.device)).float().cpu().numpy()
        camp /= np.maximum(np.linalg.norm(camp, axis=1, keepdims=True), 1e-8)
        z = lambda name, v: (v - terms[name]["mean"]) / terms[name]["std"]
        rows = []
        for i, p in enumerate(out_raw):
            ce, mse, cos = float(p[ce_i]), float(np.mean((p[top2] - source_raw[top2]) ** 2)), float(camp[i] @ target)
            rows.append({"index": i, "audiobox_ce": ce, "audiobox_pq": float(p[pq_i]), "top2_emotion_mse": mse,
                         "target_camp_cosine": cos,
                         "selection_reward": (z("audiobox_ce", ce) + z("target_camp_cosine", cos) - z("top2_emotion_mse", mse)) / 3})
        best = max(range(len(rows)), key=lambda i: (rows[i]["selection_reward"], -i))
        return best, rows

    def convert(self, source24, target24, seed, flow_steps=10, strength=1.0, gender="Off",
                candidates=1, precision="bf16"):
        """Generator: yields (partial_wave, status, report) after each section; the last yield is final."""
        infer = self.infer
        if not 1.0 <= len(source24) / 24000 <= MAX_SOURCE_S:
            raise ValueError("The performance must be 1 s to 5 min long.")
        target24 = target24[:int(REF_MAX_S * 24000)]
        if len(target24) / 24000 < REF_MIN_S:
            raise ValueError("The voice reference must be at least 3 s (5-10 s works best).")
        started = time.perf_counter()
        dev, model = self.device, self.model
        target16 = infer.resample(target24, 24000, 16000)
        target_raw = infer.ears_raw_scores([target24], self.ears, self.extractor, self.norm, dev)[0]
        target_scores = self._scale(target_raw)
        reference = infer.make_reference(model, target16, target24, dev)
        projection = model.flow.spk_embed_affine_layer
        projection.scale = float(strength)
        spans = self.sections(len(source24), 24000)
        out = np.zeros(len(source24) + 24000, np.float32)
        weight = np.zeros_like(out)
        fade = int(OVERLAP_S * 24000)
        report = {"seed": seed, "flow_steps": flow_steps, "strength": strength, "gender_boost": gender,
                  "candidates": candidates, "precision": precision, "sections": []}
        for k, (a, b) in enumerate(spans):
            section = source24[a:b]
            source_raw = infer.ears_raw_scores([section], self.ears, self.extractor, self.norm, dev)[0]
            scores = self._scale(source_raw)
            # Time-varying style/emotion from the source, stable timbre/age/gender from the target (as in infer.py).
            for name in infer.TARGET_STATIC:
                scores[self.names.index(name)] = target_scores[self.names.index(name)]
            if gender in GENDER_TARGET:
                scores[self.names.index("GEND")] = GENDER_TARGET[gender] / 6.0
            projection.condition = torch.from_numpy(scores[None]).to(dev)
            projection.present = torch.ones((1, 99), device=dev)
            tokens = infer.tokenize_waveform(model.tokenizer, infer.resample(section, 24000, 16000))
            token = torch.from_numpy(tokens[None]).to(dev)
            waves = []
            for offset in range(0, candidates, 8):
                count = min(8, candidates - offset)
                torch.manual_seed(seed + 7919 * k + 100003 * offset)
                torch.cuda.manual_seed_all(seed + 7919 * k + 100003 * offset)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    mel = model.flow_inference(speech_tokens=token.expand(count, -1).contiguous(), ref_dict=dict(reference),
                                               n_cfm_timesteps=int(flow_steps), finalize=True)
                    generated, _ = model.hift_inference(mel)
                waves += [w.reshape(-1) for w in generated.float().cpu().numpy()]
            best, rows = self._rank(waves, source_raw, reference) if candidates > 1 else (0, [])
            wave = np.nan_to_num(waves[best])[:b - a]
            # Linear crossfade over the overlap so sections join without a seam.
            ramp = np.ones(len(wave), np.float32)
            if k > 0:
                ramp[:fade] = np.linspace(0, 1, min(fade, len(wave)), dtype=np.float32)
            if k < len(spans) - 1:
                ramp[-fade:] = np.minimum(ramp[-fade:], np.linspace(1, 0, fade, dtype=np.float32))
            out[a:a + len(wave)] += wave * ramp
            weight[a:a + len(wave)] += ramp
            report["sections"].append({"start_s": a / 24000, "end_s": b / 24000, "winner": best, "candidates": rows})
            done = out[:a + len(wave)] / np.maximum(weight[:a + len(wave)], 1e-6)
            status = f"Section {k + 1} of {len(spans)} done"
            yield np.clip(done, -1, 1), status, report
        report["seconds"] = len(source24) / 24000
        report["total_wall_seconds"] = time.perf_counter() - started
        final = np.clip(out[:len(source24)] / np.maximum(weight[:len(source24)], 1e-6), -1, 1)
        yield final, "Done", report
