module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python app.py --host 127.0.0.1 --port {{port}}"
        ],
        on: [{
          // Capture the URL Gradio prints ("Running on local URL: http://127.0.0.1:PORT").
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      // Sets the local 'url' variable pinokio.js uses to light up the "Open Web UI" tab.
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
