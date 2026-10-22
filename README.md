# Humaneness Voice Converter (Pinokio)

A 1-click [Pinokio](https://pinokio.co) launcher with a Gradio GUI for
[LAION's Humaneness Voice Converter](https://huggingface.co/laion/Humaneness-Voice-Converter):
give it a **source** recording and a short clip of a **target** speaker, and it says the source
content in the target voice while keeping much of the source performance (timing, emotion).
It is audio-to-audio voice conversion, not text-to-speech.

## What you need

- An NVIDIA GPU with CUDA (the model's validated inference path refuses to run on CPU).
- About 1.5 GB for the weights plus about 6 GB for the Python environment.

## Install and use

1. In Pinokio, open this launcher and click **Install**. It creates a Python 3.13 environment,
   installs torch 2.9.1 (CUDA 12.8) and the model card's pinned stack (Chatterbox 0.1.7 with
   `--no-deps`, transformers 5.14.1), and downloads the weights into `app/humaneness-vc`.
2. Click **Start**, then **Open Web UI**.
3. **1 Performance**: upload or record what should be said, and how (1 s to 5 min), or pick a video
   file, or reuse a recent result. Performances over 20 s convert in overlapping sections with a live
   preview.
4. **2 Voice reference**: upload or record a clean 5-10 s clip of one speaker, or click **Use** on a
   voice in the **Voice library**.
5. Click **Convert voice**. The result is a mono 24 kHz WAV, saved in `app/outputs/` with the seed in
   its filename and a metrics JSON. **Try another random seed** re-runs with a new seed.

Advanced options:

- **Seed**: blank = random. Same recordings + seed + settings = the same take.
- **Conditioning strength** (default 1.0, LAION's tested value): how strongly the model emphasizes
  the voice and expression cues. Higher can sound less natural; 0 turns off only the extra emphasis.
- **Gender boost** (experimental, Off by default): pushes the model's gender conditioning toward male
  or female. Not part of LAION's release; use a matching reference and compare with Off.
- **Candidates**: 4/8/16 draws several takes per section and keeps the best by the model's proxy
  scores (content enjoyment, target-speaker similarity, emotion match). Slower.
- **Flow steps** (10 = trained) and **precision** (bf16 default, or fp32).

## Voice library and presets

- **Starter voices**: 10 voices (5 female, 5 male) from the CSTR VCTK corpus (CC BY 4.0), with tone
  names derived from the model's own timbre scores. Filter by group and by female/male.
- **Create a preset**: save the current voice reference under a name, a **group** (e.g. one group per
  film or project), a voice type and notes. Optionally remember the current strength and gender boost
  with it, so clicking **Use** restores them.
- **Manage presets and groups**: add groups, move a preset to another group, delete presets, or
  **import a folder as a group** (every audio or video file becomes a preset named after the file;
  first 10 s used).
- Presets live in `app/voices/custom/` (audio + `presets.json`). That folder is git-ignored, so your
  own voices never leave your machine.

**Update** pulls this launcher and re-syncs the weights from Hugging Face. **Reset** deletes the
environment and the downloaded weights.

## Ethics

Only convert voices you are authorized to use. Get consent where appropriate, disclose converted
speech, and never use it for impersonation, fraud or deception (see the
[model card](https://huggingface.co/laion/Humaneness-Voice-Converter)).

## API

The GUI is a standard Gradio app, so its `convert` endpoint can be called programmatically while
the app is running. Replace `PORT` with the port Pinokio shows. Inputs, in order: performance audio,
performance video (or `None`), voice reference, seed (string, `""` = random), conditioning strength,
gender boost (`"Off"`, `"Boost male"`, `"Boost female"`), candidates (`"1"`, `"4"`, `"8"`, `"16"`),
flow steps, precision. Outputs: WAV path, summary, metrics, seed used.

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:PORT/")
wav, summary, metrics, seed = client.predict(
    handle_file("source.wav"), None, handle_file("target.wav"),
    "", 1.0, "Off", "1", 10, "bf16", api_name="/convert")
print(wav, summary)
```

### JavaScript

```javascript
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:PORT/");
const result = await client.predict("/convert", [
  handle_file("source.wav"), null, handle_file("target.wav"), "", 1.0, "Off", "1", 10, "bf16"
]);
console.log(result.data);
```

### Curl

```bash
curl -X POST http://127.0.0.1:PORT/gradio_api/call/convert \
  -H "Content-Type: application/json" \
  -d '{"data": [{"path": "https://example.com/source.wav"}, null, {"path": "https://example.com/target.wav"}, "", 1.0, "Off", "1", 10, "bf16"]}'
# returns {"event_id": "..."}; then stream the result:
curl -N http://127.0.0.1:PORT/gradio_api/call/convert/EVENT_ID
```

### Command line (no GUI)

Same engine and presets as the GUI, no server needed:

```bash
cd app
env/Scripts/python cli.py --list-presets
env/Scripts/python cli.py --source take.wav --preset "My Film/Narrator" --out narrator_take.wav
env/Scripts/python cli.py --source take.mp4 --target voice_8s.wav --out out.wav --strength 1.0 --seed 42
```

## Credits and licenses

Starter voices: CSTR VCTK Corpus 0.92 by Christophe Veaux, Junichi Yamagishi and Kirsten MacDonald,
University of Edinburgh (CC BY 4.0); clips trimmed to 10 s, resampled to 24 kHz and peak-normalized.

Model, `infer.py` and conditioning by [LAION](https://huggingface.co/laion) (CC BY 4.0), built on
[Resemble AI's Chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT). This launcher only
installs and wraps them; the weights are downloaded from Hugging Face, not redistributed here.
