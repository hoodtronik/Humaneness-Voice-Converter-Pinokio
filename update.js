module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      env: {
        "UV_LINK_MODE": "copy"
      },
      message: [
        "uv pip install -r requirements.txt"
      ]
    }
  }, {
    // Re-sync the weights and infer.py with the Hugging Face repo (only changed files download).
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
  }]
}
