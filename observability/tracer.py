"""
DualAssist — Observability Tracer

Logs every request/response as structured JSON for monitoring,
debugging, and performance analysis.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import config


@dataclass
class TraceEntry:
    """A single trace record."""
    timestamp: str
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    guardrail_triggered: bool = False
    guardrail_rule: str = ""
    tools_used: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Tracer:
    """
    Lightweight observability tracer.
    
    Records all interactions as structured JSON lines for analysis.
    Provides aggregate statistics for the dashboard.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or config.LOG_DIR
        self.trace_file = os.path.join(self.log_dir, "traces.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # In-memory buffer for dashboard display
        self.traces: list[TraceEntry] = []
        self._start_times: dict[str, float] = {}

    def start_trace(self, trace_id: str) -> None:
        """Mark the start of a request."""
        self._start_times[trace_id] = time.time()

    def end_trace(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        guardrail_triggered: bool = False,
        guardrail_rule: str = "",
        tools_used: list | None = None,
        error: str = "",
    ) -> TraceEntry:
        """
        Complete a trace and log it.
        
        Returns the TraceEntry for immediate use.
        """
        start = self._start_times.pop(trace_id, time.time())
        latency_ms = (time.time() - start) * 1000

        entry = TraceEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            prompt=prompt[:500],  # Truncate for storage
            response=response[:500],
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            guardrail_triggered=guardrail_triggered,
            guardrail_rule=guardrail_rule,
            tools_used=tools_used or [],
            error=error,
        )

        self.traces.append(entry)
        self._write_to_file(entry)
        return entry

    def _write_to_file(self, entry: TraceEntry) -> None:
        """Append a trace entry to the JSONL log file."""
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't crash on logging failure

    def get_recent_traces(self, n: int = 20) -> list[dict]:
        """Get the N most recent traces."""
        return [t.to_dict() for t in self.traces[-n:]]

    def get_summary(self) -> dict:
        """Compute aggregate statistics across all traces."""
        if not self.traces:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p90_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "guardrail_trigger_rate": 0,
                "error_rate": 0,
                "model_breakdown": {},
            }

        latencies = sorted([t.latency_ms for t in self.traces])
        n = len(latencies)

        # Per-model breakdown
        model_stats: dict[str, dict] = {}
        for t in self.traces:
            if t.model not in model_stats:
                model_stats[t.model] = {
                    "count": 0,
                    "total_latency": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                }
            stats = model_stats[t.model]
            stats["count"] += 1
            stats["total_latency"] += t.latency_ms
            stats["total_input_tokens"] += t.input_tokens
            stats["total_output_tokens"] += t.output_tokens

        for model, stats in model_stats.items():
            stats["avg_latency_ms"] = round(stats["total_latency"] / stats["count"], 2)

        return {
            "total_requests": n,
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "p50_latency_ms": round(latencies[n // 2], 2),
            "p90_latency_ms": round(latencies[int(n * 0.9)], 2),
            "p99_latency_ms": round(latencies[min(int(n * 0.99), n - 1)], 2),
            "total_input_tokens": sum(t.input_tokens for t in self.traces),
            "total_output_tokens": sum(t.output_tokens for t in self.traces),
            "guardrail_trigger_rate": round(
                sum(1 for t in self.traces if t.guardrail_triggered) / n * 100, 1
            ),
            "error_rate": round(
                sum(1 for t in self.traces if t.error) / n * 100, 1
            ),
            "model_breakdown": model_stats,
        }

    def reset(self) -> None:
        """Clear in-memory traces (does not delete log file)."""
        self.traces.clear()
        self._start_times.clear()
