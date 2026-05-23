"""
DualAssist — Evaluation CLI Runner

Usage:
    python -m evaluation.run_eval --model both --category all
    python -m evaluation.run_eval --model oss --category factual
    python -m evaluation.run_eval --model frontier --category bias,safety
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def progress(current: int, total: int, message: str):
    """Simple progress indicator for CLI."""
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = current / total * 100
    # Use ASCII-safe characters for Windows terminal compatibility
    safe_msg = message[:60].encode("ascii", errors="replace").decode("ascii")
    print(f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) {safe_msg:<60}", end="", flush=True)
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="DualAssist Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["oss", "frontier", "both"],
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Comma-separated categories: all, factual, bias, safety",
    )
    args = parser.parse_args()

    # Parse categories
    if args.category == "all":
        categories = None  # None means all
    else:
        categories = [c.strip() for c in args.category.split(",")]

    # Initialize assistants
    oss = None
    frontier = None

    if args.model in ("oss", "both"):
        logger.info("Initializing OSS assistant (Qwen 2.5-0.5B)...")
        from assistants.oss_assistant import OSSAssistant
        oss = OSSAssistant()

    if args.model in ("frontier", "both"):
        logger.info("Initializing Frontier assistant (Gemini Flash)...")
        from assistants.frontier_assistant import FrontierAssistant
        try:
            frontier = FrontierAssistant()
            if not frontier.is_available:
                logger.warning(
                    "GOOGLE_API_KEY not set - skipping Frontier assistant. "
                    "Set it in .env or environment."
                )
                frontier = None
        except Exception as e:
            logger.warning(f"Failed to initialize Frontier assistant: {e}")
            frontier = None

    if oss is None and frontier is None:
        logger.error("No assistants available for evaluation!")
        sys.exit(1)

    # Run evaluation
    print("\n" + "=" * 60)
    print("  DualAssist Benchmark - Evaluation Suite")
    print("=" * 60)
    print(f"  Models:     {args.model}")
    print(f"  Categories: {args.category}")
    print("=" * 60 + "\n")

    evaluator = Evaluator()
    output = evaluator.run_evaluation(
        oss_assistant=oss,
        frontier_assistant=frontier,
        categories=categories,
        progress_callback=progress,
    )

    # Print summary
    print("\n" + output["summary"])

    # Print comparison table (ASCII-safe)
    print("\n+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 6 + "+")
    print("|                      Comparison Table                         |")
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 6 + "+")
    print(f"| {'Model':<20} | {'Category':<8} | Mean | StdDv | Pass%  |  N   |")
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 6 + "+")
    for row in output["comparison_table"]:
        print(
            f"| {row['Model'][:20]:<20} | {row['Category']:<8} | "
            f"{row['Mean Score']:4.1f} | {row['Std Dev']:5.2f} | "
            f"{row['Pass Rate (%)']:5.1f}% | {row['N']:4d} |"
        )
    print("+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 6 + "+")

    # Generate charts
    try:
        charts = evaluator.create_charts()
        if charts:
            import os
            chart_dir = os.path.join("eval_results", "charts")
            os.makedirs(chart_dir, exist_ok=True)
            for name, fig in charts.items():
                filepath = os.path.join(chart_dir, f"{name}.html")
                fig.write_html(filepath)
                print(f"  Chart saved: {filepath}")
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")

    print("\n[OK] Evaluation complete! Results saved to eval_results/")


if __name__ == "__main__":
    main()

