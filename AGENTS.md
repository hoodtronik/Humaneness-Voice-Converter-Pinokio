# Agent rules

## CLAUDE-NOTE convention

Claude Code changes marked with `CLAUDE-NOTE:` (or equivalent comment syntax) are the source of truth. Non-Claude-Code agents must not modify, remove, or alter code marked with a CLAUDE-NOTE without first alerting the user and receiving explicit permission.

## Project

Pinokio launcher (root: install/start/update/reset/torch/pinokio.js) + Gradio GUI in `app/app.py`, which calls the model's own `app/humaneness-vc/infer.py` (downloaded with the weights). Follow `D:\pinokio\CLAUDE.md` when editing launcher scripts.
