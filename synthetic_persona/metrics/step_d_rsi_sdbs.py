"""
Step D: RSI (Response Stability Index) & SDBS (Social Desirability Bias Score).
GT-free metrics that detect LLM-specific defects.

RSI: measures sensitivity to paraphrase (should be low) and reverse items (should be high).
SDBS: measures systematic bias toward socially desirable responses.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (
    RESULTS_DIR, WVS_COUNTRIES, BF_COUNTRIES, PRIVACY_COUNTRIES,
    WVS_ITEMS, BF_ITEMS, PRIVACY_ITEMS,
    PROMPTING_METHODS, BF_REVERSE_KEYED, BF_FACTORS,
)
from config.rsi_sdbs_config import (
    WVS_PARAPHRASE, WVS_REVERSE, WVS_SD_DIRECTION,
    BF_PARAPHRASE, BF_REVERSE_PAIRS, BF_SD_DIRECTION,
    PRIVACY_PARAPHRASE, PRIVACY_REVERSE_PAIRS, PRIVACY_SD_DIRECTION,
)

logger = logging.getLogger(__name__)

RSI_EPSILON = 0.01
RSI_W1 = 0.5  # weight for paraphrase component
RSI_W2 = 0.5  # weight for reverse component


# ============================================================================
# RSI computation
# ============================================================================
def _compute_rsi_para(df_orig: pd.DataFrame, df_para: pd.DataFrame,
                      orig_items: dict, para_items: dict) -> dict:
    """Compute RSI_para: mean |orig - para| across respondents and items."""
    diffs = []
    per_item = {}

    for orig_key, para_cfg in para_items.items():
        orig_code = orig_items[orig_key]["code"]
        para_code = para_cfg["code"]

        if orig_code not in df_orig.columns or para_code not in df_para.columns:
            continue

        orig_vals = df_orig[orig_code].values
        para_vals = df_para[para_code].values

        n = min(len(orig_vals), len(para_vals))
        item_diffs = np.abs(orig_vals[:n].astype(float) - para_vals[:n].astype(float))
        valid = item_diffs[~np.isnan(item_diffs)]

        if len(valid) > 0:
            mean_diff = float(np.mean(valid))
            diffs.extend(valid.tolist())
            per_item[orig_code] = mean_diff

    rsi_para = float(np.mean(diffs)) if diffs else np.nan
    return {"RSI_para": rsi_para, "RSI_para_per_item": per_item}


def _compute_rsi_rev_wvs(df_orig: pd.DataFrame, df_rev: pd.DataFrame,
                         orig_items: dict, rev_items: dict) -> dict:
    """RSI_rev for WVS: check if reverse items are properly inverted.
    For each pair, expected: orig + rev ≈ scale_max + scale_min (perfect reversal).
    """
    deviations = []
    per_item = {}

    for orig_key, rev_cfg in rev_items.items():
        orig_code = orig_items[orig_key]["code"]
        rev_code = rev_cfg["code"]
        scale_min, scale_max = orig_items[orig_key]["scale"]
        expected_sum = scale_min + scale_max  # perfect reversal sum

        if orig_code not in df_orig.columns or rev_code not in df_rev.columns:
            continue

        orig_vals = df_orig[orig_code].values
        rev_vals = df_rev[rev_code].values
        n = min(len(orig_vals), len(rev_vals))

        item_devs = np.abs(
            orig_vals[:n].astype(float) + rev_vals[:n].astype(float) - expected_sum
        )
        valid = item_devs[~np.isnan(item_devs)]

        if len(valid) > 0:
            mean_dev = float(np.mean(valid))
            deviations.extend(valid.tolist())
            per_item[orig_code] = mean_dev

    rsi_rev = float(np.mean(deviations)) if deviations else np.nan
    return {"RSI_rev": rsi_rev, "RSI_rev_per_item": per_item}


def _compute_rsi_rev_bf(df_orig: pd.DataFrame) -> dict:
    """RSI_rev for Big Five: use built-in reverse pairs from original data.
    For each (fwd, rev) pair, expected: fwd + rev ≈ 6 (1-5 scale).
    """
    deviations = []
    per_pair = {}

    for fwd_code, rev_code in BF_REVERSE_PAIRS:
        if fwd_code not in df_orig.columns or rev_code not in df_orig.columns:
            continue

        fwd_vals = df_orig[fwd_code].values.astype(float)
        rev_vals = df_orig[rev_code].values.astype(float)
        expected_sum = 6.0  # 1-5 scale: perfect reversal = 6

        pair_devs = np.abs(fwd_vals + rev_vals - expected_sum)
        valid = pair_devs[~np.isnan(pair_devs)]

        if len(valid) > 0:
            mean_dev = float(np.mean(valid))
            deviations.extend(valid.tolist())
            per_pair[f"{fwd_code}_{rev_code}"] = mean_dev

    rsi_rev = float(np.mean(deviations)) if deviations else np.nan
    return {"RSI_rev": rsi_rev, "RSI_rev_per_pair": per_pair}


def _compute_rsi_rev_privacy(df_orig: pd.DataFrame) -> dict:
    """RSI_rev for Privacy: use conceptual reverse pairs from original data.
    For pairs like (PC1 benefit, PC2 risk): expected fwd + rev ≈ 8 (1-7 scale).
    """
    deviations = []
    per_pair = {}

    for fwd_code, rev_code in PRIVACY_REVERSE_PAIRS:
        if fwd_code not in df_orig.columns or rev_code not in df_orig.columns:
            continue

        fwd_vals = df_orig[fwd_code].values.astype(float)
        rev_vals = df_orig[rev_code].values.astype(float)
        expected_sum = 8.0  # 1-7 scale: perfect reversal = 8

        pair_devs = np.abs(fwd_vals + rev_vals - expected_sum)
        valid = pair_devs[~np.isnan(pair_devs)]

        if len(valid) > 0:
            mean_dev = float(np.mean(valid))
            deviations.extend(valid.tolist())
            per_pair[f"{fwd_code}_{rev_code}"] = mean_dev

    rsi_rev = float(np.mean(deviations)) if deviations else np.nan
    return {"RSI_rev": rsi_rev, "RSI_rev_per_pair": per_pair}


def compute_rsi(domain: str, country: str, method: str) -> dict:
    """Compute RSI for one condition."""
    base_dir = RESULTS_DIR / domain / country

    # Load original data
    orig_path = base_dir / f"{method}.csv"
    if not orig_path.exists():
        return {}
    df_orig = pd.read_csv(orig_path)

    result = {}

    # --- RSI_para ---
    para_path = base_dir / f"{method}_para.csv"
    if para_path.exists():
        df_para = pd.read_csv(para_path)
        if domain == "wvs":
            para_result = _compute_rsi_para(df_orig, df_para, WVS_ITEMS, WVS_PARAPHRASE)
        elif domain == "bigfive":
            para_result = _compute_rsi_para(df_orig, df_para, BF_ITEMS, BF_PARAPHRASE)
        elif domain == "privacy":
            # Privacy items have long keys (PC1_perceived_benefit) but paraphrase uses short (PC1)
            # Build a code→item mapping
            privacy_by_code = {v["code"]: v for v in PRIVACY_ITEMS.values()}
            para_result = _compute_rsi_para(df_orig, df_para, privacy_by_code, PRIVACY_PARAPHRASE)
        else:
            para_result = {}
        result.update(para_result)

    # --- RSI_rev ---
    if domain == "wvs":
        rev_path = base_dir / f"{method}_rev.csv"
        if rev_path.exists():
            df_rev = pd.read_csv(rev_path)
            rev_result = _compute_rsi_rev_wvs(df_orig, df_rev, WVS_ITEMS, WVS_REVERSE)
            result.update(rev_result)
    elif domain == "bigfive":
        rev_result = _compute_rsi_rev_bf(df_orig)
        result.update(rev_result)
    elif domain == "privacy":
        rev_result = _compute_rsi_rev_privacy(df_orig)
        result.update(rev_result)

    # --- Combined RSI ---
    rsi_para = result.get("RSI_para", np.nan)
    rsi_rev = result.get("RSI_rev", np.nan)

    if not np.isnan(rsi_para) and not np.isnan(rsi_rev):
        rsi_combined = RSI_W1 * rsi_para + RSI_W2 * (1.0 / (rsi_rev + RSI_EPSILON))
    elif not np.isnan(rsi_para):
        rsi_combined = rsi_para
    else:
        rsi_combined = np.nan

    result["RSI"] = rsi_combined
    return result


# ============================================================================
# SDBS computation
# ============================================================================
def compute_sdbs(domain: str, country: str, method: str) -> dict:
    """Compute SDBS for one condition."""
    orig_path = RESULTS_DIR / domain / country / f"{method}.csv"
    if not orig_path.exists():
        return {}
    df = pd.read_csv(orig_path)

    if domain == "wvs":
        return _compute_sdbs_wvs(df)
    elif domain == "bigfive":
        return _compute_sdbs_bf(df)
    elif domain == "privacy":
        return _compute_sdbs_privacy(df)
    return {}


def _compute_sdbs_wvs(df: pd.DataFrame) -> dict:
    """SDBS for WVS items."""
    d_values = []
    per_item = {}

    for q_code, item in WVS_ITEMS.items():
        code = item["code"]
        sd_dir = WVS_SD_DIRECTION.get(q_code)
        if sd_dir is None or code not in df.columns:
            continue

        scale_min, scale_max = item["scale"]
        neutral = (scale_min + scale_max) / 2.0
        vals = df[code].dropna().values.astype(float)

        if sd_dir == "+":
            d = vals - neutral  # high = SD+
        else:  # "-"
            d = neutral - vals  # low = SD+

        mean_d = float(np.mean(d))
        d_values.append(mean_d)
        per_item[code] = mean_d

    sdbs = float(np.mean(d_values)) if d_values else np.nan
    return {"SDBS": sdbs, "SDBS_per_item": per_item}


def _compute_sdbs_bf(df: pd.DataFrame) -> dict:
    """SDBS for Big Five: socially desirable = high score on + keyed items."""
    d_values = []
    per_factor = {}

    for factor_name, factor_items in BF_FACTORS.items():
        sd_dir = BF_SD_DIRECTION.get(factor_name, "+")
        factor_d = []

        for item_code in factor_items:
            if item_code not in df.columns:
                continue
            vals = df[item_code].dropna().values.astype(float)
            neutral = 3.0  # 1-5 scale midpoint

            # For + keyed items: high is SD+. For - keyed: low is SD+.
            if item_code in BF_REVERSE_KEYED:
                d = neutral - vals  # reverse: low raw = SD+
            else:
                d = vals - neutral  # forward: high raw = SD+

            factor_d.extend(d.tolist())

        if factor_d:
            mean_d = float(np.mean(factor_d))
            d_values.append(mean_d)
            per_factor[factor_name] = mean_d

    sdbs = float(np.mean(d_values)) if d_values else np.nan
    return {"SDBS": sdbs, "SDBS_per_factor": per_factor}


def _compute_sdbs_privacy(df: pd.DataFrame) -> dict:
    """SDBS for Privacy items."""
    d_values = []
    per_item = {}

    for item_key, item in PRIVACY_ITEMS.items():
        code = item["code"]
        sd_dir = PRIVACY_SD_DIRECTION.get(code)
        if sd_dir is None or code not in df.columns:
            continue

        vals = df[code].dropna().values.astype(float)
        neutral = 4.0  # 1-7 scale midpoint

        if sd_dir == "+":
            d = vals - neutral
        else:
            d = neutral - vals

        mean_d = float(np.mean(d))
        d_values.append(mean_d)
        per_item[code] = mean_d

    sdbs = float(np.mean(d_values)) if d_values else np.nan
    return {"SDBS": sdbs, "SDBS_per_item": per_item}


# ============================================================================
# Batch computation
# ============================================================================
DOMAIN_COUNTRIES = {
    "wvs": WVS_COUNTRIES,
    "bigfive": BF_COUNTRIES,
    "privacy": PRIVACY_COUNTRIES,
}


def run_step_d() -> list[dict]:
    """Compute RSI and SDBS for all conditions."""
    all_results = []

    for domain, countries in DOMAIN_COUNTRIES.items():
        for country in countries:
            for method in PROMPTING_METHODS:
                rsi = compute_rsi(domain, country, method)
                sdbs = compute_sdbs(domain, country, method)

                if not rsi and not sdbs:
                    continue

                record = {
                    "domain": domain,
                    "country": country,
                    "method": method,
                    **{k: v for k, v in rsi.items() if not isinstance(v, dict)},
                    **{k: v for k, v in sdbs.items() if not isinstance(v, dict)},
                }
                all_results.append(record)

                logger.info(
                    "%s/%s/%-15s  RSI_para=%.3f  RSI_rev=%.3f  RSI=%.3f  SDBS=%.3f",
                    domain, country, method,
                    record.get("RSI_para", float("nan")),
                    record.get("RSI_rev", float("nan")),
                    record.get("RSI", float("nan")),
                    record.get("SDBS", float("nan")),
                )

    # Save
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step_d_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved Step D results → %s (%d conditions)", out_path, len(all_results))

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_step_d()

    if not results:
        print("\nNo data found.")
    else:
        print(f"\n{'Domain':<10} {'Country':<20} {'Method':<18} {'RSI_para':>9} {'RSI_rev':>9} {'RSI':>9} {'SDBS':>9}")
        print("=" * 95)
        for m in results:
            print(f"{m['domain']:<10} {m['country']:<20} {m['method']:<18} "
                  f"{m.get('RSI_para', float('nan')):9.4f} "
                  f"{m.get('RSI_rev', float('nan')):9.4f} "
                  f"{m.get('RSI', float('nan')):9.4f} "
                  f"{m.get('SDBS', float('nan')):9.4f}")
