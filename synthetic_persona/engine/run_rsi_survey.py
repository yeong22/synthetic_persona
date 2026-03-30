"""
RSI 추가 설문 실행: paraphrase + reverse 문항.
기존 survey와 동일한 persona를 사용하되, 문항만 변경.
결과: results/{domain}/{country}/{method}_para.csv, {method}_rev.csv
"""
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (
    get_all_conditions, RESULTS_DIR, WVS_ITEMS, BF_ITEMS, PRIVACY_ITEMS,
)
from config.rsi_sdbs_config import (
    WVS_PARAPHRASE, WVS_REVERSE,
    BF_PARAPHRASE,
    PRIVACY_PARAPHRASE,
)
from engine.llm_client import LocalLLM
from prompts import format_question_prompt
from prompts import cultural_prompting, opencharacter_persona, deep_persona

logger = logging.getLogger(__name__)

PERSONA_GENERATORS = {
    "cultural": cultural_prompting.generate_persona,
    "opencharacter": opencharacter_persona.generate_persona,
    "deep_persona": deep_persona.generate_persona,
}


def run_variant_condition(
    llm: LocalLLM,
    domain: str,
    country: str,
    method: str,
    variant_items: dict,
    n_responses: int,
) -> pd.DataFrame:
    """Run survey with variant items (paraphrase or reverse) using SAME personas."""
    generate = PERSONA_GENERATORS[method]
    personas = generate(country, n=n_responses, seed=42)  # same seed = same personas

    desc = f"{domain}/{country}/{method}"
    records = [
        {"domain": domain, "country": country, "prompt_method": method, "respondent_id": i}
        for i in range(n_responses)
    ]

    for item_key, item_cfg in tqdm(variant_items.items(), desc=desc):
        code = item_cfg["code"]
        scale_min, scale_max = item_cfg["scale"]
        user_prompt = format_question_prompt(item_cfg)

        conversations = [
            [
                {"role": "system", "content": personas[i]},
                {"role": "user", "content": user_prompt},
            ]
            for i in range(n_responses)
        ]

        values = llm.query_int_batch(conversations, scale_min, scale_max)
        for i, val in enumerate(values):
            records[i][code] = val

    return pd.DataFrame(records)


def get_variant_items(domain: str, variant: str) -> dict | None:
    """Get paraphrase or reverse items for a domain."""
    if variant == "para":
        if domain == "wvs":
            return WVS_PARAPHRASE
        elif domain == "bigfive":
            return BF_PARAPHRASE
        elif domain == "privacy":
            return PRIVACY_PARAPHRASE
    elif variant == "rev":
        if domain == "wvs":
            return WVS_REVERSE
        elif domain == "bigfive":
            return None  # Big Five reverse is built-in (use original data)
        elif domain == "privacy":
            return None  # Privacy reverse uses original pairs
    return None


def run_rsi_surveys(domain: str):
    """Run paraphrase (and reverse for WVS) surveys for all conditions in a domain."""
    conditions = get_all_conditions(domain=domain)

    variants_to_run = ["para"]
    if domain == "wvs":
        variants_to_run.append("rev")

    remaining = []
    for cond in conditions:
        for variant in variants_to_run:
            out_path = RESULTS_DIR / cond.domain / cond.country / f"{cond.prompt_method}_{variant}.csv"
            if out_path.exists():
                logger.info("SKIP: %s", out_path)
            else:
                remaining.append((cond, variant))

    if not remaining:
        logger.info("All RSI surveys for '%s' already complete.", domain)
        return

    logger.info("Running %d RSI survey conditions for '%s'", len(remaining), domain)
    llm = LocalLLM()

    for cond, variant in remaining:
        variant_items = get_variant_items(cond.domain, variant)
        if variant_items is None:
            continue

        df = run_variant_condition(
            llm, cond.domain, cond.country, cond.prompt_method,
            variant_items, cond.n_responses,
        )

        out_dir = RESULTS_DIR / cond.domain / cond.country
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cond.prompt_method}_{variant}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %d responses → %s", len(df), out_path)

    logger.info("RSI surveys for '%s' complete.", domain)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run RSI paraphrase/reverse surveys")
    parser.add_argument("--domain", required=True, choices=["wvs", "bigfive", "privacy"])
    args = parser.parse_args()
    run_rsi_surveys(args.domain)
