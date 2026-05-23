# DualAssist — Cost & Latency Analysis

## Model Specifications

| Metric | OSS (Qwen 2.5-0.5B) | Frontier (Gemini 2.0 Flash) |
|--------|----------------------|----------------------------|
| **Parameters** | 0.5 billion | Unknown (proprietary) |
| **Inference Mode** | Local (CPU/GPU) | Cloud API |
| **Quantization** | FP16 / auto | N/A |
| **Context Window** | 32,768 tokens | 1,048,576 tokens |

## Cost Comparison

| Cost Category | OSS (Qwen 2.5-0.5B) | Frontier (Gemini 2.0 Flash) |
|---------------|----------------------|----------------------------|
| **Per 1K input tokens** | $0.00 (self-hosted) | ~$0.10 |
| **Per 1K output tokens** | $0.00 (self-hosted) | ~$0.40 |
| **HF Spaces (free tier)** | $0.00/month | N/A |
| **HF Spaces (CPU upgrade)** | ~$5/month | N/A |
| **GPU (A10G, HF Spaces)** | ~$1.05/hr | N/A |
| **Est. cost for 1000 convos** | $0.00 (free tier) | ~$0.50-2.00 |

## Latency Analysis

### Local Development (CPU)

| Percentile | OSS (Qwen 2.5-0.5B) | Frontier (Gemini Flash) |
|------------|----------------------|------------------------|
| **P50** | ~2,000-4,000 ms | ~500-800 ms |
| **P90** | ~5,000-8,000 ms | ~800-1,200 ms |
| **P99** | ~10,000+ ms | ~1,500-2,000 ms |
| **Model load** | ~15-30s (first request) | ~0s (API) |

### HF Spaces (Free CPU Tier)

| Percentile | OSS (Qwen 2.5-0.5B) |
|------------|----------------------|
| **P50** | ~3,000-6,000 ms |
| **P90** | ~8,000-12,000 ms |
| **P99** | ~15,000+ ms |
| **Cold start** | ~60-120s |

> **Note**: Latency values are estimates based on typical hardware.
> Actual values depend on input length, system load, and hardware.

## Deployment Options

| Platform | Hardware | Est. Cost | Latency (P50) | Notes |
|----------|----------|-----------|---------------|-------|
| **HF Spaces (Free)** | 2 vCPU, 16GB RAM | Free | ~3-6s | Recommended for demo |
| **HF Spaces (CPU+)** | 8 vCPU, 32GB RAM | $5/mo | ~1-3s | Better for production |
| **Modal** | CPU instance | ~$0.01/hr | ~2-4s | Serverless, scales to zero |
| **RunPod** | A40 GPU | ~$0.39/hr | ~100-300ms | Best latency |
| **Ollama (local)** | Your machine | Free | ~1-3s | Fully private |
| **Replicate** | CPU/GPU | ~$0.001/run | ~1-5s | Pay-per-use |

## Recommendations

1. **Demo/Evaluation**: HF Spaces free tier is sufficient for the 0.5B model
2. **Production (low traffic)**: HF Spaces CPU+ or Modal for cost efficiency
3. **Production (high traffic)**: RunPod or dedicated GPU for sub-second latency
4. **Privacy-sensitive**: Ollama for fully local deployment
