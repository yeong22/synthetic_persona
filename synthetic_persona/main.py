"""
Synthetic Persona Survey — main CLI entry point.

Usage:
    python main.py survey --domain wvs        # Run survey simulation
    python main.py survey --domain privacy    # Run privacy survey
    python main.py metrics                    # Compute all metrics (Step A + C)
    python main.py analyze                    # Run rank concordance analysis
    python main.py status                     # Check pipeline status
"""
import sys
import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.experiment_config import (
    RESULTS_DIR, DATA_DIR, WVS_COUNTRIES, PRIVACY_COUNTRIES, PROMPTING_METHODS,
)


def cmd_survey(args):
    """Run survey simulation for a domain."""
    from engine.run_survey import run_domain
    run_domain(args.domain)


def cmd_metrics(args):
    """Compute Step A (GT-based) + Step C (GT-free) metrics."""
    from metrics.step_a_gt_based import run_step_a
    from metrics.step_c_compute_all import run_step_c

    print("=== Step A: GT-based metrics ===")
    run_step_a()

    print("\n=== Step C: GT-free metrics ===")
    run_step_c()


def cmd_analyze(args):
    """Run DI vs GT rank concordance analysis."""
    from metrics.analysis import run_analysis, print_summary

    results = run_analysis()
    print_summary(results["per_country"], results["pooled"])


def cmd_status(args):
    """Print pipeline status: what exists, what's missing."""
    print("=" * 60)
    print("Pipeline Status")
    print("=" * 60)

    # GT data
    print("\n[GT Data]")
    wvs_gt = DATA_DIR / "wvs_gt" / "distributions.json"
    priv_gt = DATA_DIR / "privacy_gt" / "distributions.json"
    print(f"  WVS GT:     {'OK' if wvs_gt.exists() else 'MISSING'} ({wvs_gt})")
    print(f"  Privacy GT: {'OK' if priv_gt.exists() else 'MISSING'} ({priv_gt})")

    # Synthetic results
    print("\n[Synthetic Results]")
    domains = [
        ("wvs", WVS_COUNTRIES),
        ("privacy", PRIVACY_COUNTRIES),
    ]
    for domain, countries in domains:
        found = 0
        total = len(countries) * len(PROMPTING_METHODS)
        for c in countries:
            for m in PROMPTING_METHODS:
                if (RESULTS_DIR / domain / c / f"{m}.csv").exists():
                    found += 1
        status = "OK" if found == total else f"{found}/{total}"
        print(f"  {domain}: {status}")

    # Metrics
    print("\n[Metrics]")
    for fname in ["step_a_results.json", "step_c_results.json", "analysis_results.json"]:
        p = RESULTS_DIR / "metrics" / fname
        print(f"  {fname}: {'OK' if p.exists() else 'MISSING'}")

    # Completion flags
    print("\n[Pipeline Flags]")
    for flag in [".02_done", ".03_done", ".04_done", ".05_done"]:
        p = RESULTS_DIR / flag
        print(f"  {flag}: {'SET' if p.exists() else '-'}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Persona Survey Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_survey = sub.add_parser("survey", help="Run survey simulation")
    p_survey.add_argument("--domain", required=True, choices=["wvs", "privacy"])
    p_survey.set_defaults(func=cmd_survey)

    p_metrics = sub.add_parser("metrics", help="Compute all metrics")
    p_metrics.set_defaults(func=cmd_metrics)

    p_analyze = sub.add_parser("analyze", help="Run concordance analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    p_status = sub.add_parser("status", help="Check pipeline status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
