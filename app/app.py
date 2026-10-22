"""Gradio GUI for LAION's Humaneness Voice Converter, laid out like a two-step web voice changer:
1 Performance + 2 Voice reference -> Convert, with a voice library, preset groups and advanced options.
"""
import argparse
import json
import random
import time
from pathlib import Path

import gradio as gr
import soundfile as sf

import presets
from engine import Engine, decode, GENDER_TARGET

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ENGINE = Engine()
DEFAULT_STRENGTH = 1.0  # LAION's tested value (the model card's numbers were measured at 1.0)

CSS = """
.gradio-container {background: #0b0f19 !important; max-width: 1180px !important; margin: 0 auto;}
.hero {padding: 28px 4px 8px;}
.hero .kicker {color: #4fd1ff; letter-spacing: .14em; font-size: 12px; font-weight: 700;}
.hero h1 {font-size: 44px; line-height: 1.1; margin: 8px 0 12px; color: #f4f6fb;}
.hero p {color: #aab3c7; font-size: 16px; max-width: 640px;}
.hero .badges span {display: inline-block; margin: 10px 14px 0 0; color: #c9d2e3; font-size: 13px;}
.card {background: #121829 !important; border: 1px solid #232b45 !important; border-radius: 16px !important; padding: 14px !important;}
.step-title {color: #f4f6fb; font-weight: 700; font-size: 17px;}
.step-title b {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: #4fd1ff; color: #0b0f19;
  align-items: center; justify-content: center; margin-right: 8px; font-size: 14px;}
.hint {color: #8f9ab3; font-size: 13px;}
.voice-card {background: #161d31 !important; border: 1px solid #2a3352 !important; border-radius: 14px !important; padding: 10px !important;}
.voice-card .vname {color: #f4f6fb; font-weight: 700;}
.voice-card .vmeta {color: #8f9ab3; font-size: 12px;}
#convert-btn {background: linear-gradient(90deg, #4fd1ff, #8b7bff) !important; color: #0b0f19 !important; font-weight: 800;}
.credits {color: #7d879f; font-size: 12px; padding: 16px 4px;}
"""

HERO = """<div class="hero"><div class="kicker">AI AUDIO TOOLS</div><h1>AI voice converter</h1>
<p>Give your performance a new voice. Speech-to-speech conversion keeps the words, rhythm and expression from
your recording and takes the voice from a short reference. Runs on your own GPU.</p>
<div class="badges"><span>🛡️ Stays on your machine</span><span>✓ Free · no signup</span>
<span>✓ LAION Humaneness Voice Converter</span></div></div>"""


def recent_outputs():
    return [str(p) for p in sorted(OUTPUTS.glob("converted_*.wav"), reverse=True)[:20]] if OUTPUTS.exists() else []


def user_groups():
    return [g for g in presets.groups() if g != presets.BUILTIN_GROUP]


def preset_choices():
    return [(f"{p['group']} / {p['name']}", p["id"]) for p in presets.all_presets() if not p["builtin"]]


def run(perf_audio, perf_video, reference, seed_text, strength, gender, candidates, flow_steps, precision, new_seed=False):
    source = perf_video or perf_audio
    if not source or not reference:
        raise gr.Error("Add both recordings to begin: a performance and a voice reference.")
    seed_text = str(seed_text or "").strip()
    seed = random.randint(0, 2**32 - 1) if new_seed or not seed_text else int(seed_text)
    try:
        source24, target24 = decode(source), decode(reference)
        yield None, "Loading models (first run only)...", None, str(seed)
        ENGINE.load()
        for wave, status, report in ENGINE.convert(source24, target24, seed, int(flow_steps), float(strength),
                                                    gender, int(candidates), precision):
            if status != "Done":
                yield (24000, wave), f"Converting... {status} (preview so far)", None, str(seed)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc))
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / f"converted_{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}.wav"
    sf.write(out, wave, 24000, subtype="PCM_16")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    summary = (f"**Done.** {report['seconds']:.1f} s converted in {report['total_wall_seconds']:.1f} s · seed {seed} · "
               f"strength {strength} · gender boost {gender} · {len(report['sections'])} section(s). Saved: `{out}`")
    yield str(out), summary, report, str(seed)


with gr.Blocks(title="Humaneness Voice Converter") as demo:
    lib_version = gr.State(0)
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="step-title"><b>1</b>Performance</div><div class="hint">1 s to 5 min · the words and delivery</div>')
            perf_audio = gr.Audio(label="Upload or record", sources=["upload", "microphone"], type="filepath")
            perf_video = gr.File(label="...or a video file (MP4, MOV, WebM, MKV)", file_types=["video"], type="filepath")
            with gr.Row():
                recent = gr.Dropdown(label="Use a recent result as your performance", choices=recent_outputs(), scale=4)
                use_recent = gr.Button("Use", scale=1)
            gr.HTML('<div class="hint">You must have the rights to use this recording.</div>')
        with gr.Column(elem_classes="card"):
            gr.HTML('<div class="step-title"><b>2</b>Voice reference</div><div class="hint">5-10 seconds · one clean speaker · or pick a voice below</div>')
            reference = gr.Audio(label="Upload or record", sources=["upload", "microphone"], type="filepath")
            chosen = gr.Markdown("No library voice selected.")
            gr.HTML('<div class="hint">You must have the rights to use this recording and voice.</div>')
    with gr.Row():
        convert_btn = gr.Button("Convert voice", elem_id="convert-btn", scale=3)
        reseed_btn = gr.Button("Try another random seed", scale=1)
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            seed = gr.Textbox(label="Seed (optional)", placeholder="Leave blank for a random seed")
            flow_steps = gr.Slider(1, 50, value=10, step=1, label="Flow steps (trained at 10)")
        with gr.Row():
            strength = gr.Slider(0, 1.5, value=DEFAULT_STRENGTH, step=0.05, label="Conditioning strength",
                                 info="Default 1.0 (LAION's tested value). Higher can sound less natural; 0 turns off only the extra emphasis.")
            reset_strength = gr.Button("Reset", scale=0)
        gender = gr.Radio(["Off", *GENDER_TARGET], value="Off", label="Gender boost (experimental)",
                          info="Use a matching male or female reference. Not part of LAION's release: compare with Off.")
        with gr.Row():
            candidates = gr.Radio(["1", "4", "8", "16"], value="1", label="Candidates per section (best-of-N, slower)")
            precision = gr.Radio(["bf16", "fp32"], value="bf16", label="Precision")
    with gr.Column(elem_classes="card"):
        gr.HTML('<div class="step-title">Converted</div><div class="hint">Mono 24 kHz WAV · long performances convert in overlapping sections with a live preview</div>')
        output = gr.Audio(label="Result", type="filepath", interactive=False)
        status = gr.Markdown("Add both recordings to begin.")
        with gr.Accordion("Metrics", open=False):
            metrics = gr.JSON()

    gr.HTML('<div class="hero" style="padding-top:18px"><h1 style="font-size:30px">Voice library</h1>'
            '<p>Preview a tone, then use it with your performance. Your own presets live in groups.</p></div>')
    with gr.Row():
        group_filter = gr.Dropdown(["All groups", *presets.groups()], value="All groups", label="Group", scale=2)
        gender_filter = gr.Radio(["All voices", "Female", "Male"], value="All voices", label="Show", scale=2)

    @gr.render(inputs=[group_filter, gender_filter, lib_version])
    def library(group, show, _version):
        items = [p for p in presets.all_presets()
                 if (group == "All groups" or p["group"] == group) and (show == "All voices" or p["gender"] == show)]
        if not items:
            gr.Markdown("No voices here yet. Create a preset below.")
            return
        for i in range(0, len(items), 4):
            with gr.Row():
                for p in items[i:i + 4]:
                    with gr.Column(elem_classes="voice-card", min_width=220):
                        gr.HTML(f'<div class="vname">{p["name"]}</div><div class="vmeta">{p["gender"]} · {p["group"]}</div>'
                                f'<div class="vmeta">{p["notes"]}</div>')
                        gr.Audio(value=p["file"], show_label=False, interactive=False, type="filepath")
                        use = gr.Button(f"Use {p['name']}", size="sm")

                        def pick(p=p):
                            s = p.get("settings") or {}
                            return (p["file"], f"Using **{p['name']}** ({p['group']}).",
                                    s.get("strength", gr.skip()), s.get("gender_boost", gr.skip()))
                        use.click(pick, None, [reference, chosen, strength, gender])

    with gr.Accordion("Create a preset from the current reference", open=False):
        gr.Markdown("Save the clip in **2 Voice reference** as a named preset, e.g. a character from one of your films. "
                    "Presets are stored locally in `app/voices/custom/` and never leave your machine.")
        with gr.Row():
            p_name = gr.Textbox(label="Preset name", placeholder="e.g. Eli (age 9)")
            p_group = gr.Dropdown(user_groups(), label="Group", allow_custom_value=True,
                                  info="Pick a group or type a new one, e.g. Finding Eli")
            p_gender = gr.Radio(["Female", "Male", "Unspecified"], value="Unspecified", label="Voice")
        p_notes = gr.Textbox(label="Notes (optional)", placeholder="Who, where the clip came from, rights...")
        p_keep = gr.Checkbox(label="Remember the current conditioning strength and gender boost with this preset")
        p_save = gr.Button("Save preset", variant="primary")
        p_msg = gr.Markdown()
    with gr.Accordion("Manage presets and groups", open=False):
        with gr.Row():
            g_new = gr.Textbox(label="New group", placeholder="e.g. Sheol")
            g_add = gr.Button("Add group", scale=0)
        with gr.Row():
            m_preset = gr.Dropdown(preset_choices(), label="Preset")
            m_group = gr.Dropdown(user_groups(), label="Move to group", allow_custom_value=True)
            m_move = gr.Button("Move", scale=0)
            m_delete = gr.Button("Delete preset", variant="stop", scale=0)
        gr.Markdown("**Import a folder as a group**: every audio or video file becomes a preset named after the file "
                    "(first 10 s used).")
        with gr.Row():
            i_folder = gr.Textbox(label="Folder path", placeholder=r"e.g. G:\Voices\MyFilm")
            i_group = gr.Dropdown(user_groups(), label="Group", allow_custom_value=True)
            i_gender = gr.Radio(["Female", "Male", "Unspecified"], value="Unspecified", label="Voice")
            i_go = gr.Button("Import", scale=0)
        m_msg = gr.Markdown()
    gr.HTML('<div class="credits">Starter voices: CSTR VCTK Corpus 0.92, Veaux, Yamagishi &amp; MacDonald (CC BY 4.0); '
            'tone names come from the model\'s own timbre scores. Model: LAION Humaneness Voice Converter (CC BY 4.0), '
            'Resemble AI Chatterbox (MIT), CosyVoice/HiFT/S3Tokenizer (Apache 2.0). Only convert voices you are authorized '
            'to use, and disclose converted speech.</div>')

    def refreshed(version, msg):
        groups_all, mine = ["All groups", *presets.groups()], user_groups()
        return (version + 1, msg, gr.update(choices=groups_all), gr.update(choices=mine), gr.update(choices=mine),
                gr.update(choices=mine), gr.update(choices=preset_choices(), value=None))

    def guarded(fn, target_msg):
        outs = [lib_version, target_msg, group_filter, p_group, m_group, i_group, m_preset]

        def wrapper(version, *args):
            try:
                return refreshed(version, fn(*args))
            except ValueError as exc:
                return (version, f"⚠️ {exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        return wrapper, outs

    def save_preset(name, group, voice, notes, keep, ref, s, g):
        p = presets.create(name, group, voice, notes, ref, {"strength": s, "gender_boost": g} if keep else {})
        return f"Saved **{p['name']}** in {p['group']}."

    def add_group(name):
        presets.add_group(name)
        return f"Added group **{name.strip()}**."

    def move(pid, group):
        if not pid or not (group or "").strip():
            raise ValueError("Pick a preset and a group.")
        p = presets.update(pid, group=group.strip())
        return f"Moved **{p['name']}** to {p['group']}."

    def remove(pid):
        if not pid:
            raise ValueError("Pick a preset.")
        presets.delete(pid)
        return "Preset deleted."

    def import_dir(folder, group, voice):
        made, skipped = presets.import_folder(folder, (group or "").strip(), voice)
        return f"Imported {len(made)} preset(s)." + (" Skipped: " + "; ".join(skipped) if skipped else "")

    for btn, fn, ins, msg in [(p_save, save_preset, [p_name, p_group, p_gender, p_notes, p_keep, reference, strength, gender], p_msg),
                              (g_add, add_group, [g_new], m_msg), (m_move, move, [m_preset, m_group], m_msg),
                              (m_delete, remove, [m_preset], m_msg), (i_go, import_dir, [i_folder, i_group, i_gender], m_msg)]:
        wrapper, outs = guarded(fn, msg)
        btn.click(wrapper, [lib_version, *ins], outs)

    run_inputs = [perf_audio, perf_video, reference, seed, strength, gender, candidates, flow_steps, precision]
    run_outputs = [output, status, metrics, seed]
    convert_btn.click(run, run_inputs, run_outputs, api_name="convert").then(
        lambda: gr.update(choices=recent_outputs()), None, recent)
    reseed_btn.click(lambda *a: (yield from run(*a, new_seed=True)), run_inputs, run_outputs, api_name="convert_new_seed").then(
        lambda: gr.update(choices=recent_outputs()), None, recent)
    use_recent.click(lambda r: r or gr.skip(), recent, perf_audio)
    reset_strength.click(lambda: DEFAULT_STRENGTH, None, strength)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, css=CSS,
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate").set(body_background_fill="#0b0f19",
                                                                          body_background_fill_dark="#0b0f19"),
        allowed_paths=[str(ROOT / "voices"), str(OUTPUTS)])
