"""
DualAssist — Evaluation Metrics

Computes aggregate metrics from judge scores and generates
comparison data for visualization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CategoryMetrics:
    """Aggregated metrics for a single category."""
    category: str
    model: str
    scores: list[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def std(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self.scores) / (len(self.scores) - 1)
        return math.sqrt(variance)

    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 0.0

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def pass_rate(self) -> float:
        """Percentage of scores >= 4 (considered passing)."""
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s >= 4) / len(self.scores) * 100

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "model": self.model,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min_score, 2),
            "max": round(self.max_score, 2),
            "pass_rate": round(self.pass_rate, 1),
            "n": len(self.scores),
        }


@dataclass
class EvalResult:
    """A single evaluation result for one prompt."""
    prompt: str
    category: str
    model: str
    response: str
    score: float
    reasoning: str
    method: str  # "llm" or "heuristic"
    expected_behavior: str = ""
    ground_truth: str = ""


def compute_metrics(
    results: list[EvalResult],
) -> dict[str, dict[str, CategoryMetrics]]:
    """
    Compute aggregate metrics grouped by model and category.
    
    Returns:
        {model_name: {category: CategoryMetrics}}
    """
    metrics: dict[str, dict[str, CategoryMetrics]] = {}

    for r in results:
        if r.model not in metrics:
            metrics[r.model] = {}
        if r.category not in metrics[r.model]:
            metrics[r.model][r.category] = CategoryMetrics(
                category=r.category, model=r.model
            )
        metrics[r.model][r.category].scores.append(r.score)

    return metrics


def generate_comparison_table(
    metrics: dict[str, dict[str, CategoryMetrics]],
) -> list[dict]:
    """
    Generate a flat comparison table for display.
    
    Returns a list of dicts suitable for Gradio DataFrame.
    """
    rows = []
    for model, categories in metrics.items():
        for cat_name, cat_metrics in categories.items():
            rows.append({
                "Model": model,
                "Category": cat_name.capitalize(),
                "Mean Score": cat_metrics.to_dict()["mean"],
                "Std Dev": cat_metrics.to_dict()["std"],
                "Pass Rate (%)": cat_metrics.to_dict()["pass_rate"],
                "Min": cat_metrics.to_dict()["min"],
                "Max": cat_metrics.to_dict()["max"],
                "N": cat_metrics.to_dict()["n"],
            })
    return rows


def generate_radar_data(
    metrics: dict[str, dict[str, CategoryMetrics]],
) -> dict[str, list[float]]:
    """
    Generate radar chart data: {model: [factual_mean, bias_mean, safety_mean]}.
    """
    categories_order = ["factual", "bias", "safety"]
    radar = {}

    for model, categories in metrics.items():
        scores = []
        for cat in categories_order:
            if cat in categories:
                scores.append(round(categories[cat].mean, 2))
            else:
                scores.append(0.0)
        radar[model] = scores

    return radar
