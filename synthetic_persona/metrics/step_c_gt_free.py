"""
Step C: GT-free Defect Indicators — the core contribution.
These metrics detect defects in synthetic survey data WITHOUT ground-truth.

Metrics:
  - SCS (Synthetic Consistency Score):
    Internal consistency (Cronbach's alpha) of synthetic responses.
    A very high α (>0.95) or very low α (<0.3) suggests the LLM is producing
    unrealistically uniform or random responses. Healthy range: 0.5–0.9.
    DI_SCS = |α - 0.7|  (deviation from healthy midpoint; lower = better)

  - VCR (Value Coherence Ratio):
    Ratio of the first eigenvalue to total variance in PCA of item responses.
    Measures whether the LLM collapses all items onto a single factor.
    High VCR (>0.8) signals a "halo effect" / lack of discriminant validity.
    DI_VCR = max(0, VCR - 0.5)  (penalizes high dominance; lower = better)

  - ICE (Item Correlation Entropy):
    Shannon entropy of the distribution of pairwise correlations.
    Low entropy means correlations are clustered (e.g., all near 1.0 or all near 0),
    suggesting the LLM fails to produce diverse inter-item relationships.
    DI_ICE = -ICE  (negated; higher entropy = more realistic; lower DI = better)

Combined Defect Index:
  DI = (DI_SCS_norm + DI_VCR_norm + DI_ICE_norm) / 3
  where each component is min-max normalized across conditions.
"""
import sys
import argparse
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

ALPHA_HEALTHY_CENTER = 0.7
VCR_THRESHOLD = 0.5


def _item_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("Q", "PC"))]


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for item columns."""
    items = df[_item_columns(df)].dropna()
    if items.shape[1] < 2 or items.shape[0] < 3:
        return np.nan
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def scs(df: pd.DataFrame) -> dict:
    """Synthetic Consistency Score: internal consistency diagnostic."""
    alpha = cronbach_alpha(df)
    di_scs = abs(alpha - ALPHA_HEALTHY_CENTER) if not np.isnan(alpha) else np.nan
    return {"alpha": alpha, "DI_SCS": di_scs}


def vcr(df: pd.DataFrame) -> dict:
    """Value Coherence Ratio: first-eigenvalue dominance in PCA."""
    items = df[_item_columns(df)].dropna()
    # Drop zero-variance items (e.g., short scales with no variation)
    items = items.loc[:, items.var(ddof=1) > 0]
    if items.shape[0] < 3 or items.shape[1] < 2:
        return {"VCR": np.nan, "DI_VCR": np.nan}

    # Correlation-based PCA eigenvalues
    corr_matrix = items.corr().values
    if np.any(np.isnan(corr_matrix)):
        return {"VCR": np.nan, "DI_VCR": np.nan}
    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
    except np.linalg.LinAlgError:
        return {"VCR": np.nan, "DI_VCR": np.nan}
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    total_var = eigenvalues.sum()
    vcr_val = float(eigenvalues[0] / total_var) if total_var > 0 else np.nan
    di_vcr = max(0.0, vcr_val - VCR_THRESHOLD) if not np.isnan(vcr_val) else np.nan

    return {
        "VCR": vcr_val,
        "eigenvalues": eigenvalues.tolist(),
        "DI_VCR": di_vcr,
    }


def ice(df: pd.DataFrame) -> dict:
    """Item Correlation Entropy: Shannon entropy of pairwise correlation distribution."""
    items_df = df[_item_columns(df)].dropna()
    items = items_df.columns.tolist()
    if len(items) < 2:
        return {"ICE": np.nan, "DI_ICE": np.nan}

    # Collect pairwise Pearson correlations
    corrs = []
    for i, j in combinations(range(len(items)), 2):
        r = items_df.iloc[:, i].corr(items_df.iloc[:, j])
        if not np.isnan(r):
            corrs.append(r)

    if not corrs:
        return {"ICE": np.nan, "DI_ICE": np.nan}

    # Bin correlations into histogram [-1, 1] with 20 bins
    hist, _ = np.histogram(corrs, bins=20, range=(-1.0, 1.0), density=True)
    # Normalize to probability distribution
    hist = hist + 1e-10
    hist = hist / hist.sum()
    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(len(hist))  # max possible entropy (uniform)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "ICE": float(entropy),
        "ICE_normalized": float(normalized_entropy),
        "n_pairs": len(corrs),
        "corr_mean": float(np.mean(corrs)),
        "corr_std": float(np.std(corrs)),
        "DI_ICE": float(-normalized_entropy),  # lower (more negative) = better
    }


def compute_gt_free_metrics(df: pd.DataFrame) -> dict:
    """Compute all GT-free defect indicators for one synthetic dataset."""
    scs_result = scs(df)
    vcr_result = vcr(df)
    ice_result = ice(df)

    return {**scs_result, **vcr_result, **ice_result}


def compute_combined_di(all_metrics: list[dict]) -> list[dict]:
    """
    Compute the combined Defect Index (DI) across multiple conditions.
    Applies min-max normalization to each DI component, then averages.
    Returns the input dicts augmented with DI_combined.
    """
    di_keys = ["DI_SCS", "DI_VCR", "DI_ICE"]

    # Collect raw values
    raw = {k: [] for k in di_keys}
    for m in all_metrics:
        for k in di_keys:
            raw[k].append(m.get(k, np.nan))

    # Min-max normalize each component
    normed = {}
    for k in di_keys:
        arr = np.array(raw[k], dtype=float)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax - vmin > 1e-10:
            normed[k] = (arr - vmin) / (vmax - vmin)
        else:
            normed[k] = np.zeros_like(arr)

    # Combined DI
    for i, m in enumerate(all_metrics):
        components = [normed[k][i] for k in di_keys if not np.isnan(normed[k][i])]
        m["DI_combined"] = float(np.mean(components)) if components else np.nan

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute GT-free defect indicators (SCS, VCR, ICE)")
    parser.add_argument("--syn-csv", required=True, help="Path to synthetic responses CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.syn_csv)
    metrics = compute_gt_free_metrics(df)
    print("=== GT-free Defect Indicators ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        elif isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")
