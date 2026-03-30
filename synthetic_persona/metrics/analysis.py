"""
DI ↔ GT-based rank concordance analysis.
Core research question: Do GT-free defect indicators (SCS, VCR, ICE)
rank prompting methods in the same order as GT-based metrics?

Two levels of analysis:
  1. Per-country: 3 methods ranked → do WD rank and DI rank agree?
  2. Pooled: all 18 conditions (6 countries × 3 methods) ranked → Spearman ρ

Inputs:
  - results/metrics/step_a_results.json  (GT-based: WD, JSD, KS, MeanDiff)
  - results/metrics/step_c_results.json  (GT-free: DI_SCS, DI_VCR, DI_ICE, DI_combined)
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import RESULTS_DIR, WVS_COUNTRIES, BF_COUNTRIES, PRIVACY_COUNTRIES

logger = logging.getLogger(__name__)

METRICS_DIR = RESULTS_DIR / "metrics"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_step_a(metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Load Step A (GT-based) results as DataFrame."""
    data = _load_json(metrics_dir / "step_a_results.json")
    df = pd.DataFrame(data)
    # Drop nested per_item column for ranking
    if "per_item" in df.columns:
        df = df.drop(columns=["per_item"])
    return df


def load_step_c(metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Load Step C (GT-free) results as DataFrame."""
    data = _load_json(metrics_dir / "step_c_results.json")
    return pd.DataFrame(data)


def merge_results(metrics_dir: Path = METRICS_DIR) -> pd.DataFrame:
    """Merge Step A and Step C results on (country, method)."""
    df_a = load_step_a(metrics_dir)
    df_c = load_step_c(metrics_dir)
    merged = pd.merge(df_a, df_c, on=["domain", "country", "method"], how="inner")
    return merged


# ---------------------------------------------------------------------------
# rank concordance
# ---------------------------------------------------------------------------
def spearman_concordance(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman ρ and Kendall τ between two arrays."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 3:
        return {"spearman_r": np.nan, "spearman_p": np.nan,
                "kendall_tau": np.nan, "kendall_p": np.nan, "n": n}

    sp_r, sp_p = stats.spearmanr(x, y)
    kt_tau, kt_p = stats.kendalltau(x, y)
    return {
        "spearman_r": float(sp_r), "spearman_p": float(sp_p),
        "kendall_tau": float(kt_tau), "kendall_p": float(kt_p),
        "n": n,
    }


# ---------------------------------------------------------------------------
# per-country analysis (3 methods ranked)
# ---------------------------------------------------------------------------
def per_country_analysis(df: pd.DataFrame) -> list[dict]:
    """
    For each country, rank 3 methods by GT metric (WD) and DI metric (DI_combined, DI_SCS).
    Report whether rankings match.
    """
    gt_cols = ["WD_mean", "JSD_mean", "KS_mean", "MeanDiff_mean"]
    di_cols = ["DI_combined", "DI_SCS", "DI_VCR", "DI_ICE"]

    results = []
    all_countries = WVS_COUNTRIES + BF_COUNTRIES + PRIVACY_COUNTRIES
    for country in all_countries:
        cdf = df[df["country"] == country].copy()
        if len(cdf) < 2:
            continue

        row = {"country": country, "n_methods": len(cdf)}

        # Rank each metric (ascending: lower value = rank 1 = better)
        for col in gt_cols + di_cols:
            if col in cdf.columns:
                cdf[f"{col}_rank"] = cdf[col].rank(method="min")

        # Method ranking table
        method_ranks = []
        for _, r in cdf.iterrows():
            entry = {"method": r["method"]}
            for col in gt_cols + di_cols:
                rank_col = f"{col}_rank"
                if rank_col in cdf.columns:
                    entry[f"{col}_rank"] = int(r[rank_col])
                    entry[col] = round(float(r[col]), 4) if col in r else None
            method_ranks.append(entry)
        row["method_ranks"] = method_ranks

        # Pairwise concordance: GT metric rank vs DI metric rank
        concordances = {}
        for gt in gt_cols:
            for di in di_cols:
                gt_rank = f"{gt}_rank"
                di_rank = f"{di}_rank"
                if gt_rank in cdf.columns and di_rank in cdf.columns:
                    gt_r = cdf[gt_rank].values
                    di_r = cdf[di_rank].values
                    match = np.array_equal(np.argsort(gt_r), np.argsort(di_r))
                    concordances[f"{di}_vs_{gt}"] = {
                        "rank_match": bool(match),
                    }
        row["concordances"] = concordances
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# pooled analysis (all 18 conditions)
# ---------------------------------------------------------------------------
def pooled_analysis(df: pd.DataFrame) -> dict:
    """
    Rank all conditions (6 countries × 3 methods = up to 18) by each metric.
    Compute Spearman ρ between GT and DI rankings.
    """
    gt_cols = ["WD_mean", "JSD_mean", "KS_mean", "MeanDiff_mean"]
    di_cols = ["DI_combined", "DI_SCS", "DI_VCR", "DI_ICE"]

    results = {}
    for gt in gt_cols:
        for di in di_cols:
            if gt in df.columns and di in df.columns:
                key = f"{di}_vs_{gt}"
                results[key] = spearman_concordance(
                    df[gt].values.astype(float),
                    df[di].values.astype(float),
                )
    return results


# ---------------------------------------------------------------------------
# summary printing
# ---------------------------------------------------------------------------
def print_summary(per_country: list[dict], pooled: dict):
    """Print human-readable analysis summary."""
    print("=" * 78)
    print("DI ↔ GT-based Rank Concordance Analysis")
    print("=" * 78)

    # --- Per-country method ranking tables ---
    print("\n[1] Per-country method rankings (rank 1 = best/lowest)")
    print("-" * 78)

    for row in per_country:
        print(f"\n  {row['country']} ({row['n_methods']} methods)")
        header = f"    {'Method':<18} {'WD':>8} {'JSD':>8} {'KS':>8} {'MeanDiff':>10}  |  {'DI_comb':>8} {'DI_SCS':>8}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for mr in row["method_ranks"]:
            wd_r = mr.get("WD_mean_rank", "")
            jsd_r = mr.get("JSD_mean_rank", "")
            ks_r = mr.get("KS_mean_rank", "")
            md_r = mr.get("MeanDiff_mean_rank", "")
            di_r = mr.get("DI_combined_rank", "")
            scs_r = mr.get("DI_SCS_rank", "")
            print(f"    {mr['method']:<18} {wd_r:>8} {jsd_r:>8} {ks_r:>8} {md_r:>10}  |  {di_r:>8} {scs_r:>8}")

        # Concordance check
        wd_di_match = row["concordances"].get("DI_combined_vs_WD_mean", {}).get("rank_match")
        scs_wd_match = row["concordances"].get("DI_SCS_vs_WD_mean", {}).get("rank_match")
        print(f"    → WD rank == DI_combined rank? {'YES' if wd_di_match else 'NO'}")
        print(f"    → WD rank == DI_SCS rank?      {'YES' if scs_wd_match else 'NO'}")

    # --- Pooled Spearman ρ table ---
    print(f"\n{'=' * 78}")
    print("[2] Pooled Spearman ρ (all conditions, n=18)")
    print("-" * 78)

    # Table header
    di_cols = ["DI_combined", "DI_SCS", "DI_VCR", "DI_ICE"]
    gt_cols = ["WD_mean", "JSD_mean", "KS_mean", "MeanDiff_mean"]

    print(f"  {'':20s}", end="")
    for di in di_cols:
        print(f"  {di:>14s}", end="")
    print()
    print("  " + "-" * (20 + 16 * len(di_cols)))

    for gt in gt_cols:
        print(f"  {gt:<20s}", end="")
        for di in di_cols:
            key = f"{di}_vs_{gt}"
            if key in pooled:
                r = pooled[key]["spearman_r"]
                p = pooled[key]["spearman_p"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {r:+.3f} {sig:>3s}    ", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()

    # Legend
    print(f"\n  * p<0.05  ** p<0.01  *** p<0.001")
    n = pooled.get("DI_combined_vs_WD_mean", {}).get("n", "?")
    print(f"  n = {n} conditions")


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def run_analysis(metrics_dir: Path = METRICS_DIR) -> dict:
    """Run full concordance analysis and return results."""
    df = merge_results(metrics_dir)
    logger.info("Merged %d conditions (Step A ∩ Step C)", len(df))

    per_country = per_country_analysis(df)
    pooled = pooled_analysis(df)

    results = {
        "per_country": per_country,
        "pooled": pooled,
        "n_conditions": len(df),
    }

    # Save
    out_path = metrics_dir / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved analysis → %s", out_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="DI ↔ GT rank concordance analysis")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR,
                        help="Directory containing step_a_results.json and step_c_results.json")
    args = parser.parse_args()

    try:
        results = run_analysis(args.metrics_dir)
        print_summary(results["per_country"], results["pooled"])
    except FileNotFoundError as e:
        print(f"\nMissing input: {e}")
        print("Run Step A and Step C first:")
        print("  python -m metrics.step_a_gt_based")
        print("  python -m metrics.step_c_gt_free --syn-csv <path>")
