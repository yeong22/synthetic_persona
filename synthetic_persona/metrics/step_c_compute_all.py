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
    RESULTS_DIR, WVS_COUNTRIES, BF_COUNTRIES, PRIVACY_COUNTRIES, PROMPTING_METHODS,
)
from metrics.step_c_gt_free import compute_gt_free_metrics, compute_combined_di

logger = logging.getLogger(__name__)

DOMAIN_COUNTRIES = {
    "wvs": WVS_COUNTRIES,
    "bigfive": BF_COUNTRIES,
    "privacy": PRIVACY_COUNTRIES,
}


def run_step_c() -> list[dict]:
    all_metrics = []

    for domain, countries in DOMAIN_COUNTRIES.items():
        for country in countries:
            for method in PROMPTING_METHODS:
                csv_path = RESULTS_DIR / domain / country / f"{method}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                m = compute_gt_free_metrics(df, domain=domain)
                m["domain"] = domain
                m["country"] = country
                m["method"] = method
                all_metrics.append(m)
                logger.info("%s/%s/%-15s  SCS=%.4f  VCR=%.4f  ICE=%.4f",
                            domain, country, method,
                            m.get("DI_SCS", float("nan")),
                            m.get("DI_VCR", float("nan")),
                            m.get("DI_ICE", float("nan")))

    # Combined DI (min-max normalized)
    all_metrics = compute_combined_di(all_metrics)

    # Clean non-serializable fields
    for m in all_metrics:
        m.pop("eigenvalues", None)
        # factor_alphas dict is JSON-serializable, keep it

    # Save
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_c_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Saved Step C results → %s (%d conditions)", out_path, len(all_metrics))

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_step_c()

    if not results:
        print("\nNo synthetic data found.")
    else:
        print(f"\n{'Domain':<10} {'Country':<20} {'Method':<18} {'DI_SCS':>8} {'DI_VCR':>8} {'DI_ICE':>8} {'DI_comb':>8}")
        print("=" * 92)
        for m in results:
            print(f"{m['domain']:<10} {m['country']:<20} {m['method']:<18} "
                  f"{m.get('DI_SCS', float('nan')):8.4f} {m.get('DI_VCR', float('nan')):8.4f} "
                  f"{m.get('DI_ICE', float('nan')):8.4f} {m.get('DI_combined', float('nan')):8.4f}")
