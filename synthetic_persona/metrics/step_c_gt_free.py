"""
Step C: GT-free Defect Indicators — the core contribution.
Detect defects in synthetic survey data WITHOUT ground-truth.

Metrics:
  - SCS: |α - healthy_center| (도메인별 healthy center)
  - VCR: max(0, λ1/Σλ - 0.5)
  - ICE: -H_norm(pairwise correlations)
  - DI_combined: min-max normalized average

Big Five 지원: factor 내 alpha (역코딩 반영), 50문항 전체 VCR/ICE.
"""
import sys
import argparse
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import (
    ALPHA_HEALTHY_CENTER, BF_FACTORS, BF_REVERSE_KEYED,
)

logger = logging.getLogger(__name__)

VCR_THRESHOLD = 0.5

# Item column detection per domain
ITEM_PREFIXES = {
    "wvs": ("Q",),
    "bigfive": ("EXT", "EST", "AGR", "CSN", "OPN"),
    "privacy": ("PC",),
}


def _item_columns(df: pd.DataFrame, domain: str = "wvs") -> list[str]:
    """Detect item columns based on domain prefix."""
    prefixes = ITEM_PREFIXES.get(domain, ("Q", "PC", "EXT", "EST", "AGR", "CSN", "OPN"))
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha on a prepared items DataFrame."""
    if items_df.shape[1] < 2 or items_df.shape[0] < 3:
        return np.nan
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def _reverse_score_bf(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-score Big Five items (6 - raw for reverse-keyed items)."""
    scored = df.copy()
    for col in scored.columns:
        if col in BF_REVERSE_KEYED:
            scored[col] = 6 - scored[col]
    return scored


def scs(df: pd.DataFrame, domain: str = "wvs") -> dict:
    """Synthetic Consistency Score.

    For Big Five: compute α per factor (reverse-scored), then average |α - center|.
    For WVS/Privacy: compute α on all items.
    """
    center = ALPHA_HEALTHY_CENTER.get(domain, 0.7)

    if domain == "bigfive":
        # Per-factor alpha (reverse-scored)
        factor_alphas = {}
        di_values = []
        items_df = df[_item_columns(df, domain)].dropna()
        scored = _reverse_score_bf(items_df)

        for factor_name, factor_items in BF_FACTORS.items():
            cols = [c for c in factor_items if c in scored.columns]
            if len(cols) < 2:
                continue
            alpha = cronbach_alpha(scored[cols])
            factor_alphas[factor_name] = alpha
            if not np.isnan(alpha):
                di_values.append(abs(alpha - center))

        mean_alpha = np.nanmean(list(factor_alphas.values())) if factor_alphas else np.nan
        di_scs = float(np.mean(di_values)) if di_values else np.nan
        return {"alpha": mean_alpha, "factor_alphas": factor_alphas, "DI_SCS": di_scs}
    else:
        items = df[_item_columns(df, domain)].dropna()
        alpha = cronbach_alpha(items)
        di_scs = abs(alpha - center) if not np.isnan(alpha) else np.nan
        return {"alpha": alpha, "DI_SCS": di_scs}


def vcr(df: pd.DataFrame, domain: str = "wvs") -> dict:
    """Value Coherence Ratio: first-eigenvalue dominance in PCA."""
    items = df[_item_columns(df, domain)].dropna()
    # Drop zero-variance items
    items = items.loc[:, items.var(ddof=1) > 0]
    if items.shape[0] < 3 or items.shape[1] < 2:
        return {"VCR": np.nan, "DI_VCR": np.nan}

    corr_matrix = items.corr().values
    if np.any(np.isnan(corr_matrix)):
        return {"VCR": np.nan, "DI_VCR": np.nan}
    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
    except np.linalg.LinAlgError:
        return {"VCR": np.nan, "DI_VCR": np.nan}
    eigenvalues = np.sort(eigenvalues)[::-1]
    total_var = eigenvalues.sum()
    vcr_val = float(eigenvalues[0] / total_var) if total_var > 0 else np.nan
    di_vcr = max(0.0, vcr_val - VCR_THRESHOLD) if not np.isnan(vcr_val) else np.nan

    return {"VCR": vcr_val, "DI_VCR": di_vcr}


def ice(df: pd.DataFrame, domain: str = "wvs") -> dict:
    """Item Correlation Entropy."""
    items_df = df[_item_columns(df, domain)].dropna()
    item_cols = items_df.columns.tolist()
    if len(item_cols) < 2:
        return {"ICE": np.nan, "DI_ICE": np.nan}

    corrs = []
    for i, j in combinations(range(len(item_cols)), 2):
        r = items_df.iloc[:, i].corr(items_df.iloc[:, j])
        if not np.isnan(r):
            corrs.append(r)

    if not corrs:
        return {"ICE": np.nan, "DI_ICE": np.nan}

    hist, _ = np.histogram(corrs, bins=20, range=(-1.0, 1.0), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(len(hist))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "ICE": float(entropy),
        "ICE_normalized": float(normalized_entropy),
        "n_pairs": len(corrs),
        "corr_mean": float(np.mean(corrs)),
        "corr_std": float(np.std(corrs)),
        "DI_ICE": float(-normalized_entropy),
    }


def compute_gt_free_metrics(df: pd.DataFrame, domain: str = "wvs") -> dict:
    """Compute all GT-free defect indicators for one synthetic dataset."""
    scs_result = scs(df, domain)
    vcr_result = vcr(df, domain)
    ice_result = ice(df, domain)
    return {**scs_result, **vcr_result, **ice_result}


def compute_combined_di(all_metrics: list[dict]) -> list[dict]:
    """Min-max normalize DI components across conditions, then average."""
    di_keys = ["DI_SCS", "DI_VCR", "DI_ICE"]

    raw = {k: [] for k in di_keys}
    for m in all_metrics:
        for k in di_keys:
            raw[k].append(m.get(k, np.nan))

    normed = {}
    for k in di_keys:
        arr = np.array(raw[k], dtype=float)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax - vmin > 1e-10:
            normed[k] = (arr - vmin) / (vmax - vmin)
        else:
            normed[k] = np.zeros_like(arr)

    for i, m in enumerate(all_metrics):
        components = [normed[k][i] for k in di_keys if not np.isnan(normed[k][i])]
        m["DI_combined"] = float(np.mean(components)) if components else np.nan

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute GT-free defect indicators")
    parser.add_argument("--syn-csv", required=True)
    parser.add_argument("--domain", default="wvs", choices=["wvs", "bigfive", "privacy"])
    args = parser.parse_args()

    df = pd.read_csv(args.syn_csv)
    metrics = compute_gt_free_metrics(df, domain=args.domain)
    print("=== GT-free Defect Indicators ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        elif isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")
