"""
Step B: Structural Fidelity Score (SFS).
Evaluates whether the synthetic data preserves the inter-item correlation structure
of the real data WITHOUT requiring exact distributional match.

Sub-metrics:
  - SignF (Sign Fidelity):  fraction of item pairs where correlation sign matches
  - SigF (Significance Fidelity): fraction of significant correlations in real data
                                   that are also significant in synthetic data
  - NullF (Null Fidelity): fraction of non-significant correlations in real data
                            that remain non-significant in synthetic data
  - SFS = mean(SignF, SigF, NullF)
"""
import sys
import argparse
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.experiment_config import RESULTS_DIR

logger = logging.getLogger(__name__)
SIG_THRESHOLD = 0.05


def _item_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("Q", "PC"))]


def _corr_with_pvalue(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson correlation matrix and p-value matrix."""
    items = _item_columns(df)
    data = df[items].dropna()
    n = len(items)
    corr_mat = np.zeros((n, n))
    pval_mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            r, p = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
            corr_mat[i, j] = corr_mat[j, i] = r
            pval_mat[i, j] = pval_mat[j, i] = p
        corr_mat[i, i] = 1.0
        pval_mat[i, i] = 0.0

    return (
        pd.DataFrame(corr_mat, index=items, columns=items),
        pd.DataFrame(pval_mat, index=items, columns=items),
    )


def sign_fidelity(corr_syn: pd.DataFrame, corr_real: pd.DataFrame) -> float:
    """Fraction of off-diagonal pairs where correlation signs match."""
    items = corr_syn.columns.tolist()
    pairs = list(combinations(range(len(items)), 2))
    if not pairs:
        return np.nan
    match = sum(
        1 for i, j in pairs
        if np.sign(corr_syn.iloc[i, j]) == np.sign(corr_real.iloc[i, j])
    )
    return match / len(pairs)


def significance_fidelity(pval_syn: pd.DataFrame, pval_real: pd.DataFrame) -> float:
    """Of pairs significant in real data, fraction also significant in synthetic."""
    items = pval_syn.columns.tolist()
    pairs = list(combinations(range(len(items)), 2))
    sig_real = [(i, j) for i, j in pairs if pval_real.iloc[i, j] < SIG_THRESHOLD]
    if not sig_real:
        return np.nan  # no significant correlations in real data
    match = sum(1 for i, j in sig_real if pval_syn.iloc[i, j] < SIG_THRESHOLD)
    return match / len(sig_real)


def null_fidelity(pval_syn: pd.DataFrame, pval_real: pd.DataFrame) -> float:
    """Of pairs NOT significant in real data, fraction also not significant in synthetic."""
    items = pval_syn.columns.tolist()
    pairs = list(combinations(range(len(items)), 2))
    null_real = [(i, j) for i, j in pairs if pval_real.iloc[i, j] >= SIG_THRESHOLD]
    if not null_real:
        return np.nan
    match = sum(1 for i, j in null_real if pval_syn.iloc[i, j] >= SIG_THRESHOLD)
    return match / len(null_real)


def compute_sfs(syn: pd.DataFrame, real: pd.DataFrame) -> dict:
    """Compute Structural Fidelity Score and sub-metrics."""
    corr_syn, pval_syn = _corr_with_pvalue(syn)
    corr_real, pval_real = _corr_with_pvalue(real)

    signf = sign_fidelity(corr_syn, corr_real)
    sigf = significance_fidelity(pval_syn, pval_real)
    nullf = null_fidelity(pval_syn, pval_real)

    components = [v for v in [signf, sigf, nullf] if not np.isnan(v)]
    sfs = float(np.mean(components)) if components else np.nan

    return {
        "SignF": signf,
        "SigF": sigf,
        "NullF": nullf,
        "SFS": sfs,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute Structural Fidelity Score")
    parser.add_argument("--syn-csv", required=True)
    parser.add_argument("--real-csv", required=True)
    args = parser.parse_args()

    syn = pd.read_csv(args.syn_csv)
    real = pd.read_csv(args.real_csv)
    metrics = compute_sfs(syn, real)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")
