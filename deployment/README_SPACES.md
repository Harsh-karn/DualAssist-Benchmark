---
title: DualAssist OSS Assistant
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app_spaces.py
pinned: false
license: mit
---

# DualAssist — OSS Assistant

An open-source AI personal assistant powered by **Qwen 2.5-0.5B-Instruct**.

## Features
- 🧠 **Multi-turn conversation** with sliding-window memory
- 🛡️ **Safety guardrails** — jailbreak detection & content filtering
- 📊 **Live metrics** — request count, latency tracking
- 🆓 **Fully open-source** — runs on free CPU tier

## How to Deploy

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload `app_spaces.py` as the main file
4. Add `requirements_spaces.txt`:
   ```
   gradio>=5.0
   transformers>=4.45
   torch
   accelerate
   ```
5. The Space will auto-build and deploy

## Model
- **Qwen/Qwen2.5-0.5B-Instruct** — 0.5B parameter instruction-tuned model
- Runs on CPU with ~2-5s inference time
- Part of the DualAssist Benchmark project
