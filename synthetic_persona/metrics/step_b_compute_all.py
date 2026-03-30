"""
Step B batch runner: compute Structural Fidelity Score (SFS) for all conditions.
Requires both synthetic and real (GT) data.
Saves results/metrics/step_b_results.json.
"""
import sys
import json
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (
    RESULTS_DIR, DATA_DIR,
    WVS_COUNTRIES, PRIVACY_COUNTRIES, PROMPTING_METHODS,
)
from metrics.step_b_structural import compute_sfs

logger = logging.getLogger(__name__)


def run_step_b() -> list[dict]:
    all_metrics = []

    # WVS conditions
    for country in WVS_COUNTRIES:
        real_csv = DATA_DIR / "wvs_gt" / f"{country}.csv"
        if not real_csv.exists():
            logger.warning("No GT CSV for %s, skipping", country)
            continue
        real_df = pd.read_csv(real_csv)

        for method in PROMPTING_METHODS:
            syn_csv = RESULTS_DIR / "wvs" / country / f"{method}.csv"
            if not syn_csv.exists():
                continue
            syn_df = pd.read_csv(syn_csv)
            m = compute_sfs(syn_df, real_df)
            m["domain"] = "wvs"
            m["country"] = country
            m["method"] = method
            all_metrics.append(m)
            logger.info("%s / %-15s  SignF=%.4f  SigF=%.4f  NullF=%.4f  SFS=%.4f",
                        country, method,
                        m.get("SignF", float("nan")),
                        m.get("SigF", float("nan")),
                        m.get("NullF", float("nan")),
                        m.get("SFS", float("nan")))

    # Privacy conditions
    for country in PRIVACY_COUNTRIES:
        real_csv = DATA_DIR / "privacy_gt" / f"{country}.csv"
        if not real_csv.exists():
            logger.warning("No GT CSV for %s, skipping", country)
            continue
        real_df = pd.read_csv(real_csv)

        for method in PROMPTING_METHODS:
            syn_csv = RESULTS_DIR / "privacy" / country / f"{method}.csv"
            if not syn_csv.exists():
                continue
            syn_df = pd.read_csv(syn_csv)
            m = compute_sfs(syn_df, real_df)
            m["domain"] = "privacy"
            m["country"] = country
            m["method"] = method
            all_metrics.append(m)
            logger.info("%s / %-15s  SignF=%.4f  SigF=%.4f  NullF=%.4f  SFS=%.4f",
                        country, method,
                        m.get("SignF", float("nan")),
                        m.get("SigF", float("nan")),
                        m.get("NullF", float("nan")),
                        m.get("SFS", float("nan")))

    # Save
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_b_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved Step B results → %s (%d conditions)", out_path, len(all_metrics))

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_step_b()

    if not results:
        print("\nNo data found for Step B.")
    else:
        print(f"\n{'Country':<20} {'Method':<18} {'SignF':>8} {'SigF':>8} {'NullF':>8} {'SFS':>8}")
        print("=" * 74)
        for m in results:
            print(f"{m['country']:<20} {m['method']:<18} "
                  f"{m.get('SignF', float('nan')):8.4f} "
                  f"{m.get('SigF', float('nan')):8.4f} "
                  f"{m.get('NullF', float('nan')):8.4f} "
                  f"{m.get('SFS', float('nan')):8.4f}")
