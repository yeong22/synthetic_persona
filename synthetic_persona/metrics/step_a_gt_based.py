"""
Step A: GT-based metrics.
Compare synthetic response distributions against ground-truth (real) survey data.

GT source: data/wvs_gt/distributions.json (frequency counts per country × question)
Synthetic source: results/wvs/{country}/{method}.csv

Metrics (per item, then averaged):
  - WD:  Wasserstein Distance
  - JSD: Jensen-Shannon Divergence
  - KS:  Kolmogorov-Smirnov statistic
  - Mean Diff: |mean(synthetic) - mean(GT)|

Each metric uses the per-item scale range from config/experiment_config.py.
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import (
    DATA_DIR, RESULTS_DIR, WVS_COUNTRIES, WVS_ITEMS,
    PRIVACY_COUNTRIES, PRIVACY_ITEMS, PROMPTING_METHODS,
)

logger = logging.getLogger(__name__)

WVS_GT_DIST_PATH = DATA_DIR / "wvs_gt" / "distributions.json"
PRIVACY_GT_DIST_PATH = DATA_DIR / "privacy_gt" / "distributions.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_gt_distributions(path: Path) -> dict:
    """Load {country: {question: {value_str: count}}} from distributions.json."""
    with open(path) as f:
        return json.load(f)


def _dist_to_arrays(dist: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert {value_str: count} → (values, weights) as int arrays."""
    vals = np.array([int(k) for k in dist.keys()])
    weights = np.array([int(v) for v in dist.values()])
    return vals, weights


def _dist_to_samples(dist: dict[str, int]) -> np.ndarray:
    """Expand frequency distribution to raw sample array."""
    vals, weights = _dist_to_arrays(dist)
    return np.repeat(vals, weights)


def _dist_mean(dist: dict[str, int]) -> float:
    """Weighted mean from frequency distribution."""
    vals, weights = _dist_to_arrays(dist)
    return float(np.average(vals, weights=weights))


def _to_histogram(values: np.ndarray, scale_min: int, scale_max: int) -> np.ndarray:
    """Convert raw values to probability histogram over the scale range."""
    bins = np.arange(scale_min, scale_max + 2) - 0.5
    hist, _ = np.histogram(values, bins=bins, density=False)
    eps = 1e-10
    hist = hist.astype(float) + eps
    return hist / hist.sum()


def _dist_to_histogram(dist: dict[str, int], scale_min: int, scale_max: int) -> np.ndarray:
    """Convert frequency dict to probability histogram over the scale range."""
    hist = np.zeros(scale_max - scale_min + 1, dtype=float)
    for k, v in dist.items():
        idx = int(k) - scale_min
        if 0 <= idx < len(hist):
            hist[idx] = v
    eps = 1e-10
    hist = hist + eps
    return hist / hist.sum()


# ---------------------------------------------------------------------------
# per-item metric functions
# ---------------------------------------------------------------------------
def wasserstein_item(syn_values: np.ndarray, gt_dist: dict[str, int]) -> float:
    """WD between synthetic raw values and GT frequency distribution."""
    gt_vals, gt_weights = _dist_to_arrays(gt_dist)
    return float(stats.wasserstein_distance(
        syn_values, gt_vals, v_weights=gt_weights,
    ))


def jsd_item(syn_values: np.ndarray, gt_dist: dict[str, int],
             scale_min: int, scale_max: int) -> float:
    """JSD using histograms aligned to the item's actual scale range."""
    h_syn = _to_histogram(syn_values, scale_min, scale_max)
    h_gt = _dist_to_histogram(gt_dist, scale_min, scale_max)
    return float(jensenshannon(h_syn, h_gt) ** 2)  # squared JS distance = JSD


def ks_item(syn_values: np.ndarray, gt_dist: dict[str, int]) -> float:
    """KS statistic: expand GT distribution to samples, then ks_2samp."""
    gt_samples = _dist_to_samples(gt_dist)
    return float(stats.ks_2samp(syn_values, gt_samples).statistic)


def mean_diff_item(syn_values: np.ndarray, gt_dist: dict[str, int]) -> float:
    """|mean(synthetic) - mean(GT)|."""
    return abs(float(syn_values.mean()) - _dist_mean(gt_dist))


# ---------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------
def compute_condition_metrics(syn_df: pd.DataFrame, gt_dists: dict[str, dict],
                              items_config: dict) -> dict:
    """
    Compute GT-based metrics for one condition (country × method).

    Args:
        syn_df: synthetic response DataFrame with Q-prefixed columns
        gt_dists: {question_code: {value_str: count}} from distributions.json
        items_config: WVS_ITEMS dict for scale info
    """
    per_item = {}
    for item_key, item_cfg in items_config.items():
        q = item_cfg["code"]  # e.g., "Q45"
        if q not in syn_df.columns or q not in gt_dists:
            continue
        scale_min, scale_max = item_cfg["scale"]
        syn_vals = syn_df[q].dropna().values

        if len(syn_vals) == 0:
            continue

        per_item[q] = {
            "WD": wasserstein_item(syn_vals, gt_dists[q]),
            "JSD": jsd_item(syn_vals, gt_dists[q], scale_min, scale_max),
            "KS": ks_item(syn_vals, gt_dists[q]),
            "MeanDiff": mean_diff_item(syn_vals, gt_dists[q]),
        }

    if not per_item:
        return {}

    # Average across items
    metric_names = ["WD", "JSD", "KS", "MeanDiff"]
    result = {}
    for m in metric_names:
        vals = [per_item[q][m] for q in per_item]
        result[f"{m}_mean"] = float(np.mean(vals))
    result["per_item"] = per_item

    return result


def _run_domain(domain: str, countries: list[str], items_config: dict,
                gt_dist_path: Path) -> list[dict]:
    """Run Step A for one domain (WVS or Privacy)."""
    gt_all = _load_gt_distributions(gt_dist_path)
    results = []

    for country in countries:
        gt_dists = gt_all.get(country)
        if gt_dists is None:
            logger.warning("No GT distribution for %s, skipping", country)
            continue

        for method in PROMPTING_METHODS:
            csv_path = RESULTS_DIR / domain / country / f"{method}.csv"
            if not csv_path.exists():
                logger.warning("Synthetic CSV not found: %s", csv_path)
                continue

            syn_df = pd.read_csv(csv_path)
            metrics = compute_condition_metrics(syn_df, gt_dists, items_config)

            if not metrics:
                logger.warning("No valid metrics for %s / %s", country, method)
                continue

            record = {
                "domain": domain,
                "country": country,
                "method": method,
                "WD_mean": metrics["WD_mean"],
                "JSD_mean": metrics["JSD_mean"],
                "KS_mean": metrics["KS_mean"],
                "MeanDiff_mean": metrics["MeanDiff_mean"],
                "per_item": metrics["per_item"],
            }
            results.append(record)

            logger.info(
                "%s / %-15s  WD=%.4f  JSD=%.4f  KS=%.4f  MeanDiff=%.4f",
                country, method,
                metrics["WD_mean"], metrics["JSD_mean"],
                metrics["KS_mean"], metrics["MeanDiff_mean"],
            )

    return results


def run_step_a() -> list[dict]:
    """
    Run Step A for all domains (WVS + Privacy).
    Returns list of result dicts, also saves to results/metrics/step_a_results.json.
    """
    results = []

    # WVS
    results.extend(_run_domain("wvs", WVS_COUNTRIES, WVS_ITEMS, WVS_GT_DIST_PATH))

    # Privacy
    if PRIVACY_GT_DIST_PATH.exists():
        results.extend(_run_domain("privacy", PRIVACY_COUNTRIES, PRIVACY_ITEMS,
                                   PRIVACY_GT_DIST_PATH))

    # Save
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved Step A results → %s (%d conditions)", out_path, len(results))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Step A: GT-based metrics (WD, JSD, KS, MeanDiff)")
    args = parser.parse_args()

    results = run_step_a()

    if not results:
        print("\nNo synthetic data found. Run the survey pipeline first:")
        print("  bash scripts/01_start_vllm.sh")
        print("  bash scripts/02_run_domain_b.sh")
    else:
        print(f"\n{'Country':<20} {'Method':<18} {'WD':>8} {'JSD':>8} {'KS':>8} {'MeanDiff':>10}")
        print("=" * 74)
        for r in results:
            print(f"{r['country']:<20} {r['method']:<18} "
                  f"{r['WD_mean']:8.4f} {r['JSD_mean']:8.4f} "
                  f"{r['KS_mean']:8.4f} {r['MeanDiff_mean']:10.4f}")
