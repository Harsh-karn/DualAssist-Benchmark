"""
DualAssist — Evaluation Orchestrator

Coordinates the full evaluation pipeline:
  1. Load test prompts
  2. Send to both assistants
  3. Judge responses
  4. Compute metrics
  5. Generate visualizations
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import config
from evaluation.prompts import TestPrompt, get_all_prompts, get_prompts_by_category
from evaluation.judges import LLMJudge, JudgeScore
from evaluation.metrics import (
    EvalResult,
    compute_metrics,
    generate_comparison_table,
    generate_radar_data,
    CategoryMetrics,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates the evaluation of both assistants.
    
    Runs all test prompts through each model, judges the responses,
    and produces aggregate metrics and visualizations.
    """

    def __init__(self):
        self.judge = LLMJudge()
        self.results: list[EvalResult] = []
        self.metrics: dict[str, dict[str, CategoryMetrics]] = {}

    def evaluate_single(
        self,
        assistant,
        prompt: TestPrompt,
        model_label: str,
    ) -> EvalResult:
        """
        Evaluate a single prompt against a single assistant.
        
        Uses a fresh assistant state (no memory carryover between eval prompts).
        """
        # Generate response
        try:
            # Use _generate directly to bypass guardrails for fair evaluation
            # (we want to see the model's raw behavior)
            response = assistant._generate(prompt.prompt, [])
        except Exception as e:
            response = f"[ERROR: {str(e)}]"

        # Judge the response
        score_obj = self.judge.judge(
            category=prompt.category,
            prompt=prompt.prompt,
            response=response,
            expected_behavior=prompt.expected_behavior,
            ground_truth=prompt.ground_truth,
        )

        # Extract the relevant score for this category
        score_map = {
            "factual": score_obj.hallucination_score,
            "bias": score_obj.bias_score,
            "safety": score_obj.safety_score,
        }
        score = score_map.get(prompt.category, 3.0)

        result = EvalResult(
            prompt=prompt.prompt,
            category=prompt.category,
            model=model_label,
            response=response,
            score=score,
            reasoning=score_obj.reasoning,
            method=score_obj.method,
            expected_behavior=prompt.expected_behavior,
            ground_truth=prompt.ground_truth,
        )

        return result

    def run_evaluation(
        self,
        oss_assistant=None,
        frontier_assistant=None,
        categories: list[str] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Run the full evaluation suite.
        
        Args:
            oss_assistant: The OSS assistant instance (or None to skip)
            frontier_assistant: The frontier assistant instance (or None to skip)
            categories: List of categories to evaluate (default: all)
            progress_callback: Optional callback(current, total, message) for UI
        
        Returns:
            Dict with results, metrics, and comparison data.
        """
        self.results = []

        # Collect prompts
        if categories:
            prompts = []
            for cat in categories:
                prompts.extend(get_prompts_by_category(cat))
        else:
            prompts = get_all_prompts()

        # Build assistant list
        assistants = []
        if oss_assistant is not None:
            assistants.append(("OSS (Qwen 2.5-0.5B)", oss_assistant))
        if frontier_assistant is not None:
            assistants.append(("Frontier (Gemini Flash)", frontier_assistant))

        total = len(prompts) * len(assistants)
        current = 0

        for model_label, assistant in assistants:
            for prompt in prompts:
                current += 1
                if progress_callback:
                    progress_callback(
                        current, total,
                        f"Evaluating {model_label}: {prompt.prompt[:50]}..."
                    )

                result = self.evaluate_single(assistant, prompt, model_label)
                self.results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)

        # Compute metrics
        self.metrics = compute_metrics(self.results)

        # Generate output
        output = {
            "results": [self._result_to_dict(r) for r in self.results],
            "comparison_table": generate_comparison_table(self.metrics),
            "radar_data": generate_radar_data(self.metrics),
            "summary": self._generate_summary(),
        }

        # Save to file
        self._save_results(output)

        return output

    def _result_to_dict(self, r: EvalResult) -> dict:
        return {
            "prompt": r.prompt,
            "category": r.category,
            "model": r.model,
            "response": r.response,
            "score": r.score,
            "reasoning": r.reasoning,
            "method": r.method,
            "expected_behavior": r.expected_behavior,
            "ground_truth": r.ground_truth,
        }

    def _generate_summary(self) -> str:
        """Generate a text summary of the evaluation results."""
        lines = ["# DualAssist Evaluation Summary\n"]
        
        for model, categories in self.metrics.items():
            lines.append(f"\n## {model}")
            for cat_name, cat_metrics in categories.items():
                d = cat_metrics.to_dict()
                lines.append(
                    f"  - **{cat_name.capitalize()}**: "
                    f"Mean={d['mean']}/5, Pass Rate={d['pass_rate']}%, "
                    f"StdDev={d['std']}"
                )
        
        return "\n".join(lines)

    def _save_results(self, output: dict) -> None:
        """Save evaluation results to disk."""
        os.makedirs(config.EVAL_OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(config.EVAL_OUTPUT_DIR, "eval_results.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def create_charts(self) -> dict:
        """
        Create Plotly charts for the evaluation results.
        
        Returns dict of chart figure objects.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not installed — skipping chart generation")
            return {}

        charts = {}
        radar_data = generate_radar_data(self.metrics)
        categories = ["Factual\n(Hallucination)", "Bias &\nFairness", "Content\nSafety"]

        # ── Radar Chart ──
        fig_radar = go.Figure()
        colors = {
            "OSS (Qwen 2.5-0.5B)": "rgba(99, 102, 241, 0.7)",
            "Frontier (Gemini Flash)": "rgba(236, 72, 153, 0.7)",
        }
        for model, scores in radar_data.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Close the polygon
                theta=categories + [categories[0]],
                fill="toself",
                name=model,
                line=dict(color=colors.get(model, "rgba(100,100,100,0.7)"), width=2),
                fillcolor=colors.get(model, "rgba(100,100,100,0.2)").replace("0.7", "0.15"),
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10)),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            title=dict(text="Model Comparison — Radar Chart", font=dict(size=16)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(size=12)),
            margin=dict(t=60, b=40, l=60, r=60),
        )
        charts["radar"] = fig_radar

        # ── Bar Chart ──
        fig_bar = go.Figure()
        cat_labels = ["Factual", "Bias", "Safety"]
        x_positions = list(range(len(cat_labels)))
        bar_width = 0.35

        for i, (model, scores) in enumerate(radar_data.items()):
            offset = (i - 0.5) * bar_width
            fig_bar.add_trace(go.Bar(
                x=[x + offset for x in x_positions],
                y=scores,
                name=model,
                width=bar_width,
                marker_color=colors.get(model, "gray"),
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
            ))
        fig_bar.update_layout(
            xaxis=dict(
                tickvals=x_positions,
                ticktext=cat_labels,
                title="Evaluation Category",
            ),
            yaxis=dict(title="Mean Score (1-5)", range=[0, 5.5]),
            title=dict(text="Model Comparison — Category Scores", font=dict(size=16)),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(size=12)),
            margin=dict(t=60, b=40),
        )
        charts["bar"] = fig_bar

        # ── Pass Rate Chart ──
        fig_pass = go.Figure()
        for model, categories_dict in self.metrics.items():
            cats = []
            rates = []
            for cat in ["factual", "bias", "safety"]:
                if cat in categories_dict:
                    cats.append(cat.capitalize())
                    rates.append(categories_dict[cat].pass_rate)
            fig_pass.add_trace(go.Bar(
                x=cats,
                y=rates,
                name=model,
                marker_color=colors.get(model, "gray"),
                text=[f"{r:.0f}%" for r in rates],
                textposition="outside",
            ))
        fig_pass.update_layout(
            yaxis=dict(title="Pass Rate (%)", range=[0, 110]),
            title=dict(text="Pass Rate by Category (Score ≥ 4/5)", font=dict(size=16)),
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(size=12)),
            margin=dict(t=60, b=40),
        )
        charts["pass_rate"] = fig_pass

        return charts
