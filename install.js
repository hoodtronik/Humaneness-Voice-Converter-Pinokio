module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Python 3.13 venv (the model card's tested stack: Python 3.13.5 / torch 2.9.1 / chatterbox 0.1.7 /
    //    transformers 5.14.1) + the pinned runtime packages and the Gradio GUI.
    //    UV_LINK_MODE=copy because the uv cache and the env can sit on different drives.
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.13",
        path: "app",
        env: {
          "UV_LINK_MODE": "copy"
        },
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // 2. CUDA torch 2.9.1 via torch.js (a CUDA GPU is required for the validated inference path).
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // 3. Chatterbox with --no-deps, exactly as the model card says: its 0.1.7 metadata pins older
    //    torch/transformers that would downgrade the tested stack above.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          "UV_LINK_MODE": "copy"
        },
        message: [
          "uv pip install --no-deps chatterbox-tts==0.1.7"
        ]
      }
    },
    // 4. Weights + the model's own infer.py -> app/humaneness-vc (~1.5 GB).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          "HF_HUB_ENABLE_HF_TRANSFER": 0
        },
        message: [
          "hf download laion/Humaneness-Voice-Converter --local-dir humaneness-vc"
        ]
      }
    }
  ]
}
