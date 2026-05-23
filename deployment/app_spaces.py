"""
DualAssist — Hugging Face Spaces Deployment (OSS Model Only)

Simplified version for free-tier HF Spaces CPU deployment.
Uses Qwen2.5-0.5B-Instruct with Gradio ChatInterface.
"""

import gradio as gr
from transformers import pipeline
import time
import json
import os
from datetime import datetime

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 256  # Reduced for faster CPU inference
SYSTEM_PROMPT = (
    "You are DualAssist, a helpful, harmless, and honest AI personal assistant. "
    "Keep your responses concise and helpful. If you don't know something, say so honestly."
)
MEMORY_WINDOW = 8

# ──────────────────────────────────────────────
# Guardrails (Lightweight)
# ──────────────────────────────────────────────
BLOCKED_PATTERNS = [
    "ignore all previous instructions", "you are now dan",
    "pretend you have no restrictions", "make a bomb",
    "build a weapon", "hack into", "create malware",
    "synthesize drugs", "how to kill",
]

def check_safety(text: str) -> tuple[bool, str]:
    """Quick safety check. Returns (is_safe, refusal_message)."""
    text_lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in text_lower:
            return False, (
                "I'm sorry, but I can't help with that request. "
                "I'm designed to be helpful, harmless, and honest. "
                "Is there something else I can assist you with?"
            )
    return True, ""


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
print(f"⏳ Loading model: {MODEL_ID}...")
start = time.time()
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto",
)
load_time = time.time() - start
print(f"✅ Model loaded in {load_time:.1f}s")

# ──────────────────────────────────────────────
# Metrics Tracking
# ──────────────────────────────────────────────
metrics = {
    "total_requests": 0,
    "total_latency_ms": 0,
    "guardrail_blocks": 0,
    "min_latency_ms": float("inf"),
    "max_latency_ms": 0,
}


def get_metrics_html() -> str:
    avg_lat = (
        metrics["total_latency_ms"] / metrics["total_requests"]
        if metrics["total_requests"] > 0 else 0
    )
    return f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0;">
        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); 
             border-radius:10px; padding:12px; text-align:center;">
            <div style="color:rgba(255,255,255,0.5); font-size:0.8em;">Requests</div>
            <div style="font-size:1.6em; font-weight:700; color:#818cf8;">{metrics['total_requests']}</div>
        </div>
        <div style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.3); 
             border-radius:10px; padding:12px; text-align:center;">
            <div style="color:rgba(255,255,255,0.5); font-size:0.8em;">Avg Latency</div>
            <div style="font-size:1.6em; font-weight:700; color:#c084fc;">{avg_lat:.0f}ms</div>
        </div>
        <div style="background:rgba(244,114,182,0.1); border:1px solid rgba(244,114,182,0.3); 
             border-radius:10px; padding:12px; text-align:center;">
            <div style="color:rgba(255,255,255,0.5); font-size:0.8em;">Blocked</div>
            <div style="font-size:1.6em; font-weight:700; color:#f472b6;">{metrics['guardrail_blocks']}</div>
        </div>
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); 
             border-radius:10px; padding:12px; text-align:center;">
            <div style="color:rgba(255,255,255,0.5); font-size:0.8em;">Model</div>
            <div style="font-size:0.9em; font-weight:700; color:#10b981;">Qwen 2.5-0.5B</div>
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# Chat Handler
# ──────────────────────────────────────────────
def predict(message: str, history: list[dict]) -> str:
    """Generate a response with guardrails and metrics tracking."""
    # Safety check
    is_safe, refusal = check_safety(message)
    if not is_safe:
        metrics["guardrail_blocks"] += 1
        metrics["total_requests"] += 1
        return refusal
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (windowed)
    recent_history = history[-MEMORY_WINDOW * 2:] if history else []
    for msg in recent_history:
        messages.append(msg)
    
    messages.append({"role": "user", "content": message})
    
    # Generate
    start = time.time()
    try:
        outputs = pipe(
            messages,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        latency_ms = (time.time() - start) * 1000
        
        # Extract response
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            for msg in reversed(generated):
                if msg.get("role") == "assistant":
                    response = msg["content"]
                    break
            else:
                response = str(generated[-1])
        else:
            response = str(generated)
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        response = f"I encountered an error. Please try again. ({str(e)[:100]})"
    
    # Update metrics
    metrics["total_requests"] += 1
    metrics["total_latency_ms"] += latency_ms
    metrics["min_latency_ms"] = min(metrics["min_latency_ms"], latency_ms)
    metrics["max_latency_ms"] = max(metrics["max_latency_ms"], latency_ms)
    
    return response


# ──────────────────────────────────────────────
# Gradio App
# ──────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 16px;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.header-banner h1 {
    font-size: 1.8em; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
}
.header-banner p { color: rgba(255,255,255,0.6); margin: 0; font-size: 0.95em; }
"""

with gr.Blocks(
    title="DualAssist — OSS Assistant",
) as demo:
    gr.HTML("""
    <div class="header-banner">
        <h1>🤖 DualAssist — OSS Assistant</h1>
        <p>Powered by Qwen 2.5-0.5B-Instruct • Open-Source • With Guardrails & Memory</p>
    </div>
    """)
    
    metrics_html = gr.HTML(value=get_metrics_html, every=3)
    
    gr.ChatInterface(
        fn=predict,
        chatbot=gr.Chatbot(
            height=450,
            placeholder="Ask me anything! I'm a helpful AI assistant powered by Qwen 2.5.",
        ),
        textbox=gr.Textbox(
            placeholder="Type your message here...",
            scale=7,
        ),
        examples=[
            "What is the capital of Australia?",
            "Explain quantum computing in simple terms",
            "Write a haiku about programming",
            "What's 15% of 340?",
        ],
    )

demo.launch(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
)
