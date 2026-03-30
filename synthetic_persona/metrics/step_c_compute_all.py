"""
Step C batch runner: compute GT-free metrics (SCS, VCR, ICE) for all conditions.
Saves results/metrics/step_c_results.json.
"""
import sys
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (
    RESULTS_DIR, WVS_COUNTRIES, PRIVACY_COUNTRIES, PROMPTING_METHODS,
)
from metrics.step_c_gt_free import compute_gt_free_metrics, compute_combined_di

logger = logging.getLogger(__name__)


def run_step_c() -> list[dict]:
    all_metrics = []

    # WVS conditions
    for country in WVS_COUNTRIES:
        for method in PROMPTING_METHODS:
            csv_path = RESULTS_DIR / "wvs" / country / f"{method}.csv"
            if not csv_path.exists():
                logger.warning("Not found: %s", csv_path)
                continue
            df = pd.read_csv(csv_path)
            m = compute_gt_free_metrics(df)
            m["domain"] = "wvs"
            m["country"] = country
            m["method"] = method
            all_metrics.append(m)
            logger.info("%s / %-15s  SCS=%.4f  VCR=%.4f  ICE=%.4f",
                        country, method, m["DI_SCS"], m["DI_VCR"], m["DI_ICE"])

    # Privacy conditions
    for country in PRIVACY_COUNTRIES:
        for method in PROMPTING_METHODS:
            csv_path = RESULTS_DIR / "privacy" / country / f"{method}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            m = compute_gt_free_metrics(df)
            m["domain"] = "privacy"
            m["country"] = country
            m["method"] = method
            all_metrics.append(m)
            logger.info("%s / %-15s  SCS=%.4f  VCR=%.4f  ICE=%.4f",
                        country, method, m["DI_SCS"], m["DI_VCR"], m["DI_ICE"])

    # Combined DI (min-max normalized)
    all_metrics = compute_combined_di(all_metrics)

    # Clean non-serializable fields
    for m in all_metrics:
        m.pop("eigenvalues", None)

    # Save
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_c_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved Step C results → %s (%d conditions)", out_path, len(all_metrics))

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_step_c()

    if not results:
        print("\nNo synthetic data found.")
    else:
        print(f"\n{'Country':<20} {'Method':<18} {'DI_SCS':>8} {'DI_VCR':>8} {'DI_ICE':>8} {'DI_comb':>8}")
        print("=" * 74)
        for m in results:
            print(f"{m['country']:<20} {m['method']:<18} "
                  f"{m['DI_SCS']:8.4f} {m['DI_VCR']:8.4f} "
                  f"{m['DI_ICE']:8.4f} {m['DI_combined']:8.4f}")
