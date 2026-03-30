"""
Survey simulation runner.
vLLM offline 추론 — 서버 불필요, 프로세스 종료 시 GPU 자동 해제.

각 condition = (domain, country, prompt_method) → n_responses personas.
각 persona가 전체 items (6문항)에 응답. 배치 추론으로 item별 300건 일괄 처리.
"""
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (
    ExperimentCondition, get_all_conditions, RESULTS_DIR,
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


def run_condition(llm: LocalLLM, cond: ExperimentCondition) -> pd.DataFrame:
    """Run one experiment condition using batch inference per item."""
    generate = PERSONA_GENERATORS[cond.prompt_method]
    personas = generate(cond.country, n=cond.n_responses, seed=42)

    desc = f"{cond.domain}/{cond.country}/{cond.prompt_method}"

    # Initialize records
    records = [
        {
            "domain": cond.domain,
            "country": cond.country,
            "prompt_method": cond.prompt_method,
            "respondent_id": resp_idx,
        }
        for resp_idx in range(cond.n_responses)
    ]

    # Batch inference per item (6 items × 300 prompts per batch)
    for item_key, item_cfg in tqdm(cond.items.items(), desc=desc):
        code = item_cfg["code"]
        scale_min, scale_max = item_cfg["scale"]
        user_prompt = format_question_prompt(item_cfg)

        # Build all conversations for this item
        conversations = [
            [
                {"role": "system", "content": personas[resp_idx]},
                {"role": "user", "content": user_prompt},
            ]
            for resp_idx in range(cond.n_responses)
        ]

        # Batch query → 300 responses at once
        values = llm.query_int_batch(conversations, scale_min, scale_max)

        for resp_idx, val in enumerate(values):
            records[resp_idx][code] = val

    return pd.DataFrame(records)


def save_results(df: pd.DataFrame, cond: ExperimentCondition):
    """Save condition results to CSV."""
    out_dir = RESULTS_DIR / cond.domain / cond.country
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cond.prompt_method}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved %d responses → %s", len(df), path)


def run_domain(domain: str):
    """Run all conditions for a specific domain."""
    conditions = [c for c in get_all_conditions() if c.domain == domain]

    # Skip already-completed conditions (resume support)
    remaining = []
    for cond in conditions:
        csv_path = RESULTS_DIR / cond.domain / cond.country / f"{cond.prompt_method}.csv"
        if csv_path.exists():
            logger.info("SKIP (already exists): %s/%s/%s",
                        cond.domain, cond.country, cond.prompt_method)
        else:
            remaining.append(cond)

    if not remaining:
        logger.info("All %d conditions for domain '%s' already complete.",
                    len(conditions), domain)
        return

    logger.info("Running %d conditions for domain '%s' (%d skipped)",
                len(remaining), domain, len(conditions) - len(remaining))

    # Load model once for all conditions — freed on function return
    llm = LocalLLM()

    for cond in remaining:
        df = run_condition(llm, cond)
        save_results(df, cond)

    logger.info("Domain '%s' complete.", domain)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run synthetic persona survey simulation")
    parser.add_argument("--domain", required=True, choices=["wvs", "privacy"],
                        help="Domain to run")
    args = parser.parse_args()
    run_domain(args.domain)
