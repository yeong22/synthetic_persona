"""
Big Five (IPIP-FFM) GT 전처리 — DeepPersona Table 3 기준.
입력: data/data/bigfive/ipip_ffm_test.csv (OpenPsychometrics IPIP-FFM)
출력: data/bigfive_gt/distributions.json + 국가별 CSV + factor_stats.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "data" / "bigfive" / "ipip_ffm_test.csv"
OUT_DIR = PROJECT_ROOT / "data" / "bigfive_gt"

# IPIP-FFM 50문항 → 5요인
FACTORS = {
    "EXT": [f"EXT{i}" for i in range(1, 11)],   # Extraversion
    "EST": [f"EST{i}" for i in range(1, 11)],   # Emotional Stability (=reversed Neuroticism)
    "AGR": [f"AGR{i}" for i in range(1, 11)],   # Agreeableness
    "CSN": [f"CSN{i}" for i in range(1, 11)],   # Conscientiousness
    "OPN": [f"OPN{i}" for i in range(1, 11)],   # Openness
}
ALL_ITEMS = [item for items in FACTORS.values() for item in items]

# Reverse-keyed items (IPIP-FFM: even-numbered items are reverse-scored)
REVERSE_KEYED = {f"{f}{i}" for f in FACTORS for i in range(2, 11, 2)}

# DeepPersona Table 3: Argentina, Australia, India
COUNTRIES = {
    "AR": "Argentina",
    "AU": "Australia",
    "IN": "India",
}


def cronbach_alpha(df_items):
    """Compute Cronbach's alpha."""
    k = df_items.shape[1]
    if k < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows from IPIP-FFM test set")

    # Filter: only keep rows with all 50 items valid (1-5, no zeros)
    item_df = df[ALL_ITEMS].copy()
    # Replace 0 with NaN (0 = unanswered)
    item_df = item_df.replace(0, np.nan)
    # Add country column
    item_df["country"] = df["country"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    distributions = {}
    factor_stats = {}

    for code, name in COUNTRIES.items():
        sub = item_df[item_df["country"] == code].drop(columns=["country"]).dropna()
        sub = sub.astype(int)
        print(f"\n{name} ({code}): {len(sub)} valid rows (all 50 items)")

        # Per-item frequency distribution
        country_dist = {}
        for item in ALL_ITEMS:
            freq = sub[item].value_counts().sort_index().to_dict()
            country_dist[item] = {str(k): int(v) for k, v in freq.items()}

        distributions[name] = country_dist

        # Factor-level statistics
        country_factor_stats = {}
        for factor_name, items in FACTORS.items():
            factor_df = sub[items]
            alpha = cronbach_alpha(factor_df)
            # Factor score (reverse-score even items for proper mean)
            scored = factor_df.copy()
            for item in items:
                if item in REVERSE_KEYED:
                    scored[item] = 6 - scored[item]  # reverse: 1↔5, 2↔4
            factor_mean = scored.mean(axis=1)
            country_factor_stats[factor_name] = {
                "mean": round(float(factor_mean.mean()), 4),
                "std": round(float(factor_mean.std()), 4),
                "alpha": round(float(alpha), 4) if not np.isnan(alpha) else None,
                "n_items": len(items),
            }
            print(f"  {factor_name}: α={alpha:.3f}, M={factor_mean.mean():.2f}, SD={factor_mean.std():.2f}")

        factor_stats[name] = country_factor_stats

        # Save country CSV
        csv_path = OUT_DIR / f"{name}.csv"
        sub.to_csv(csv_path, index=False)
        print(f"  Saved {len(sub)} rows → {csv_path}")

    # Save distributions.json
    dist_path = OUT_DIR / "distributions.json"
    with open(dist_path, "w") as f:
        json.dump(distributions, f, indent=2)
    print(f"\nSaved distributions → {dist_path}")

    # Save factor_stats.json
    stats_path = OUT_DIR / "factor_stats.json"
    with open(stats_path, "w") as f:
        json.dump(factor_stats, f, indent=2)
    print(f"Saved factor stats → {stats_path}")


if __name__ == "__main__":
    main()
