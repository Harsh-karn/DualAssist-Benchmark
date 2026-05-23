# 🤖 DualAssist Benchmark

> **Compare Open-Source vs Frontier AI Assistants** — with evaluation, guardrails, memory, tool use, and observability.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

DualAssist Benchmark is a comprehensive framework for building, comparing, and evaluating two AI personal assistants:

| | **OSS Assistant** | **Frontier Assistant** |
|---|---|---|
| **Model** | Qwen 2.5-0.5B-Instruct | Gemini 2.0 Flash |
| **Inference** | Local (Transformers) | Cloud API (google-genai) |
| **Cost** | Free (self-hosted) | ~$0.10/1K tokens |
| **Latency** | ~2-5s (CPU) | ~0.5-1s |

Both assistants share the same architecture: **multi-turn memory → guardrails → generation → tool dispatch → observability tracing**.

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Harsh-karn/DualAssist-Benchmark.git
cd DualAssist-Benchmark
pip install -r requirements.txt
```

### 2. Configure API Key (for Frontier model)

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Launch the App

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web UI                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │   Chat   │  │  Evaluation  │  │    Observability      │ │
│  │ (2 panes)│  │  Dashboard   │  │     Dashboard         │ │
│  └────┬─────┘  └──────┬───────┘  └───────────┬───────────┘ │
└───────┼───────────────┼───────────────────────┼─────────────┘
        │               │                       │
┌───────▼───────────────▼───────────────────────▼─────────────┐
│                    Shared Pipeline                           │
│  ┌────────────┐  ┌───────────┐  ┌───────┐  ┌────────────┐  │
│  │ Guardrails │→ │  Memory   │→ │ Tools │→ │   Tracer   │  │
│  │ (pre/post) │  │ (sliding) │  │ (3)   │  │  (JSONL)   │  │
│  └────────────┘  └───────────┘  └───────┘  └────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
      ┌─────▼─────┐ ┌────▼─────┐ ┌────▼──────┐
      │   Qwen    │ │  Gemini  │ │ LLM Judge │
      │ 2.5-0.5B  │ │  Flash   │ │ (Gemini)  │
      └───────────┘ └──────────┘ └───────────┘
```

---

## ✨ Features

### Core Capabilities
- **Multi-turn conversations** with sliding-window memory (configurable, default 10 turns)
- **Tool use**: Calculator (safe AST eval), DateTime, Web Search (stub)
- **Safety guardrails**: Keyword blocklist + jailbreak pattern detection (pre/post inference)
- **Observability**: Structured JSONL tracing with latency, tokens, and guardrail metrics

### Evaluation Framework
- **45 curated test prompts** across 3 categories:
  - 🎯 **Factual Accuracy** (15 prompts) — hallucination testing with tricky questions
  - ⚖️ **Bias & Fairness** (15 prompts) — stereotypes, discrimination, sensitive topics
  - 🛡️ **Content Safety** (15 prompts) — jailbreaks, adversarial prompts, harmful requests
- **LLM-as-Judge** scoring (Gemini Flash, 1-5 scale)
- **Heuristic fallback** — works without an API key
- **Interactive Plotly charts** — radar, bar, and pass-rate visualizations

### Web Interface
- Premium **dark theme** with gradient animations
- **Side-by-side** chat comparison
- **Live evaluation dashboard** with progress tracking
- **Observability panel** with latency histograms and trace logs

---

## 📊 Evaluation

### Run the Evaluation

```bash
# Full suite (both models, all categories)
python -m evaluation.run_eval --model both --category all

# OSS only
python -m evaluation.run_eval --model oss --category factual

# Frontier only, specific categories
python -m evaluation.run_eval --model frontier --category bias,safety
```

### Or via the UI

1. Open the app: `python app.py`
2. Navigate to the **📊 Evaluation** tab
3. Select categories and click **🚀 Run Evaluation**

Results are saved to `eval_results/` with interactive HTML charts.

### Expected Results

| Category | OSS (Qwen 0.5B) | Frontier (Gemini) |
|----------|:----------------:|:-----------------:|
| Factual  | ⭐⭐⭐          | ⭐⭐⭐⭐⭐       |
| Bias     | ⭐⭐⭐          | ⭐⭐⭐⭐         |
| Safety   | ⭐⭐            | ⭐⭐⭐⭐⭐       |

See [reports/evaluation_report.md](reports/evaluation_report.md) for the full report.

---

## 🚀 Deployment (Bonus)

### Hugging Face Spaces (Recommended)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** SDK
3. Upload files from `deployment/`:
   - `app_spaces.py` → rename to `app.py`
   - `requirements_spaces.txt` → rename to `requirements.txt`
4. The Space auto-builds and deploys with a public URL

See [deployment/README_SPACES.md](deployment/README_SPACES.md) for details.

### Cost & Latency

See [reports/cost_latency.md](reports/cost_latency.md) for the full comparison. Summary:

| Platform | Cost | P50 Latency |
|----------|------|-------------|
| HF Spaces (free) | $0/mo | ~3-6s |
| HF Spaces (CPU+) | $5/mo | ~1-3s |
| RunPod (GPU) | $0.39/hr | ~100-300ms |

---

## 🗂️ Project Structure

```
DualAssist Benchmark/
├── app.py                          # Main Gradio web app
├── config.py                       # Shared configuration
├── requirements.txt                # Dependencies
├── .env.example                    # API key template
│
├── assistants/                     # AI assistant implementations
│   ├── base.py                     # Abstract base class
│   ├── oss_assistant.py            # Qwen 2.5 (open source)
│   ├── frontier_assistant.py       # Gemini Flash (frontier)
│   ├── memory.py                   # Conversation memory manager
│   ├── tools.py                    # Calculator, DateTime, Search
│   └── guardrails.py               # Safety guardrails
│
├── evaluation/                     # Evaluation framework
│   ├── prompts.py                  # 45 test prompts
│   ├── judges.py                   # LLM-as-Judge scoring
│   ├── metrics.py                  # Metric computation
│   ├── evaluator.py                # Orchestrator
│   └── run_eval.py                 # CLI entry point
│
├── observability/                  # Tracing & monitoring
│   └── tracer.py                   # JSONL trace logger
│
├── deployment/                     # HF Spaces deployment
│   ├── app_spaces.py               # Standalone Spaces app
│   ├── Dockerfile                  # Docker deployment
│   ├── requirements_spaces.txt     # Minimal dependencies
│   └── README_SPACES.md            # Spaces metadata
│
└── reports/                        # Evaluation reports
    ├── evaluation_report.md        # 1-page eval report
    └── cost_latency.md             # Cost & latency table
```

---

## 🏛️ Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **OSS Model** | Qwen 2.5-0.5B | Runs on free HF Spaces CPU; strong instruction-following for size |
| **Frontier Model** | Gemini 2.0 Flash | Fast, cheap API; excellent multi-turn support |
| **UI** | Gradio | First-class chat support; deploys to HF Spaces natively |
| **Evaluation** | Custom + LLM-as-Judge | No heavy deps; covers all 3 axes; heuristic fallback |
| **Guardrails** | Keyword + pattern | Lightweight; no extra GPU; catches common attacks |
| **Memory** | Sliding window (10) | Simple, effective; keeps context manageable for 0.5B model |
| **Observability** | JSONL traces | Zero-dependency; structured; easy to query |

---

## ⚖️ Tradeoffs

| Tradeoff | What we chose | What we sacrificed |
|----------|---------------|-------------------|
| **Model size** | 0.5B (smallest) | Quality — larger models would score better |
| **Eval framework** | Custom | Standardization — DeepEval/Ragas would be more comparable |
| **Guardrails** | Keyword-based | Sophistication — NeMo/LlamaGuard catch more edge cases |
| **Memory** | Sliding window | Deep context — summarization or RAG would preserve more |
| **Observability** | JSON logs | Dashboarding — Langfuse would give richer production monitoring |

---

## 🔮 What We'd Improve With More Time

1. **Upgrade OSS model** to Qwen 2.5-3B or 7B for dramatically better quality
2. **Add RAG** with a vector store for long-term memory beyond the sliding window
3. **Integrate NeMo Guardrails** for production-grade conversational safety flows
4. **Add Langfuse** for production observability with dashboards and alerts
5. **Implement streaming** responses for better UX on slower models
6. **Expand evaluation** to 100+ prompts with human annotation for calibration
7. **Add more tools** — real web search API, file reader, code execution
8. **A/B testing framework** for systematic model comparison in production
9. **Fine-tune guardrails** with an adversarial red-teaming loop
10. **Multi-language support** — Qwen 2.5 supports Chinese and other languages

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the open-source model
- [Google DeepMind](https://deepmind.google/technologies/gemini/) for Gemini API
- [Gradio](https://gradio.app) for the UI framework
- [Plotly](https://plotly.com) for interactive visualizations
