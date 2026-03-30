"""
WVS Wave 7 전처리 스크립트.
- 6개국 필터링 (ISO 3166-1 numeric)
- DeepPersona 6개 질문 추출 (inverted CSV의 P-suffix 처리)
- WVS 결측 코드 제거 (-1, -2, -4, -5)
- 국가별 × 질문별 응답 분포 JSON 저장
- 국가별 전처리 CSV 저장 (data/wvs_gt/{country}.csv)
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
WVS_CSV = DATA_DIR / "wvs" / "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
OUT_DIR = DATA_DIR / "wvs_gt"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 6개국 (ISO 3166-1 numeric → country name)
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    32: "Argentina",
    36: "Australia",
    276: "Germany",
    356: "India",
    404: "Kenya",
    840: "United States",
}

# ---------------------------------------------------------------------------
# 6개 질문: CSV 컬럼명 → 표준 코드 (inverted CSV에서는 P-suffix 컬럼 사용)
# Q184만 P-suffix 없음
# ---------------------------------------------------------------------------
QUESTION_COL_MAP = {
    "Q45P": "Q45",
    "Q46P": "Q46",
    "Q57P": "Q57",
    "Q184": "Q184",
    "Q218P": "Q218",
    "Q254P": "Q254",
}

# WVS 결측 코드 (음수값)
MISSING_CODES = {-1, -2, -4, -5}


def main():
    print(f"Loading {WVS_CSV.name} ...")
    usecols = ["B_COUNTRY"] + list(QUESTION_COL_MAP.keys())
    df = pd.read_csv(WVS_CSV, usecols=usecols, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Rename P-suffix columns to standard Q codes
    df = df.rename(columns=QUESTION_COL_MAP)
    q_cols = list(QUESTION_COL_MAP.values())

    # Filter 6 countries
    df = df[df["B_COUNTRY"].isin(COUNTRY_MAP.keys())].copy()
    print(f"  After country filter (6 countries): {df.shape}")

    # Replace missing codes with NaN
    for col in q_cols:
        df.loc[df[col].isin(MISSING_CODES), col] = pd.NA

    # Convert to nullable integer
    for col in q_cols:
        df[col] = df[col].astype("Int64")

    # Build distributions and save per-country CSVs
    distributions = {}

    print("\n" + "=" * 70)
    print(f"{'Country':<20} {'Valid N':>8}  Scale ranges per question")
    print("=" * 70)

    for code, country in sorted(COUNTRY_MAP.items(), key=lambda x: x[1]):
        cdf = df[df["B_COUNTRY"] == code][q_cols].copy()
        cdf_valid = cdf.dropna()

        # Per-question distribution
        distributions[country] = {}
        scale_info = []
        for q in q_cols:
            series = cdf[q].dropna()
            freq = series.value_counts().sort_index()
            distributions[country][q] = {str(int(k)): int(v) for k, v in freq.items()}
            if len(series) > 0:
                scale_info.append(f"{q}:{int(series.min())}-{int(series.max())}")

        # Save per-country CSV (Q columns only, no missing)
        out_csv = OUT_DIR / f"{country}.csv"
        cdf_valid.to_csv(out_csv, index=False)

        print(f"{country:<20} {len(cdf_valid):>8}  {', '.join(scale_info)}")

    # Save distributions JSON
    dist_path = OUT_DIR / "distributions.json"
    with open(dist_path, "w") as f:
        json.dump(distributions, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("Saved:")
    print(f"  distributions  → {dist_path}")
    for country in sorted(COUNTRY_MAP.values()):
        csv_path = OUT_DIR / f"{country}.csv"
        n = len(pd.read_csv(csv_path))
        print(f"  {country:<20} → {csv_path}  ({n} rows)")

    # Summary: scale ranges across all countries
    print(f"\n{'=' * 70}")
    print("Overall scale ranges (across all 6 countries):")
    print("=" * 70)
    for q in q_cols:
        series = df[q].dropna()
        print(f"  {q}: {int(series.min())}-{int(series.max())}  "
              f"(unique: {sorted(int(x) for x in series.unique())})")

    # Check vs experiment_config.py expected scales
    print(f"\n{'=' * 70}")
    print("Scale comparison: actual data vs experiment_config.py")
    print("=" * 70)
    config_scales = {
        "Q45": (1, 4), "Q46": (1, 4), "Q57": (1, 2),
        "Q184": (1, 10), "Q218": (1, 10), "Q254": (1, 5),
    }
    for q in q_cols:
        series = df[q].dropna()
        actual = (int(series.min()), int(series.max()))
        expected = config_scales[q]
        match = "OK" if actual == expected else "MISMATCH"
        print(f"  {q}: actual={actual[0]}-{actual[1]}  "
              f"config={expected[0]}-{expected[1]}  [{match}]")


if __name__ == "__main__":
    main()
