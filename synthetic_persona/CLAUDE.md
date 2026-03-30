# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Synthetic Persona Survey Research**: GT-free defect indicators (SCS, VCR, ICE) for evaluating LLM-generated synthetic survey data quality without ground-truth.

Research question: Can we detect defects in synthetic persona survey responses without access to real human data?

## Two Sub-projects

### 1. Synthetic Persona Survey (main project, root-level)
- **Model**: Qwen/Qwen2.5-3B-Instruct via vLLM offline mode (tensor parallel=2, CUDA_VISIBLE_DEVICES=0,1)
- **Domains**: WVS Wave 7 (6 countries x 6 items) + Privacy Calculus (2 countries x 6 items)
- **Prompting**: 3 strategies — Cultural Prompting (Tao et al. 2024), OpenCharacter persona, DeepPersona
- **Key metric**: DI (Defect Index) = min-max normalized average of DI_SCS, DI_VCR, DI_ICE
- **24 total conditions**: (6 WVS countries + 2 Privacy countries) x 3 prompting methods, 300 responses each

### 2. Autoresearch (`autoresearch/` subdir)
Autonomous LLM pretraining research (Karpathy). Independent sub-project with its own pyproject.toml. See `autoresearch/README.md`.

## Setup

```bash
# Requires Python 3.11+, NVIDIA GPU (RTX 3090 x 2+ recommended)
uv sync          # preferred
# or: pip install -e .
```

## Commands

```bash
# Full pipeline (env check -> surveys -> metrics -> analysis)
# No server management needed — vLLM loads model in-process, GPU freed on exit
bash run_all.sh

# Or step by step:
bash scripts/01_start_vllm.sh        # Environment check (vLLM import + model cache)
bash scripts/02_run_domain_b.sh       # WVS survey simulation (loads model, runs, frees GPU)
bash scripts/03_run_domain_a.sh       # Privacy Calculus simulation
bash scripts/04_compute_metrics.sh    # All metrics (GT-based + structural + GT-free)
bash scripts/05_analyze.sh            # DI<->GT rank concordance analysis

# main.py CLI (alternative to shell scripts for individual steps)
python main.py survey --domain wvs       # Run WVS survey simulation
python main.py survey --domain privacy   # Run Privacy survey simulation
python main.py metrics                   # Compute Step A (GT-based) + Step C (GT-free)
python main.py analyze                   # Run rank concordance analysis
python main.py status                    # Check pipeline status (what exists, what's missing)

# Run individual modules directly (prefix with CUDA_VISIBLE_DEVICES=0,1)
CUDA_VISIBLE_DEVICES=0,1 python -m engine.run_survey --domain wvs
python -m metrics.step_c_gt_free --syn-csv results/wvs/Argentina/cultural.csv
python -m metrics.step_a_gt_based [--syn-dir results/wvs/]
python -m metrics.step_c_compute_all     # Batch GT-free metrics for all conditions
python -m metrics.step_b_structural --syn-csv <path> --real-csv <path>
python -m metrics.analysis --metrics-dir results/metrics

# Test modules
python -m engine.llm_client              # Parse tests + inference test (GPU needed)
python -m engine.llm_client --no-gpu     # Parse tests only (no GPU)
python -m prompts.deep_persona           # Print sample personas
python -m prompts.opencharacter_persona
python -m prompts.cultural_prompting
```

## Architecture

### Data Flow
```
experiment_config.py (24 conditions: country x method)
  -> prompts/*.py generate_persona() -> list of system prompts
  -> engine/run_survey.py iterates conditions, batch inference per item (300 prompts/batch)
  -> engine/llm_client.py LocalLLM.query_int_batch() -> list of integer Likert responses
  -> results/{domain}/{country}/{method}.csv
  -> metrics/step_c_compute_all.py (batch) or step_c_gt_free.py (single CSV)
  -> metrics/step_a_gt_based.py (if GT available in data/wvs_gt/)
  -> metrics/step_b_structural.py (if both syn + real CSVs)
  -> metrics/analysis.py (Spearman/Kendall rank concordance: DI vs GT)
```

### Key Modules
```
config/experiment_config.py    — Central config: countries, items (with per-item scales), model settings, ExperimentCondition dataclass
main.py                        — CLI entry point (survey/metrics/analyze/status subcommands)
prompts/__init__.py            — Shared: format_question_prompt(), run_survey() — question formatting + per-persona survey execution
prompts/cultural_prompting.py  — Simplest: all n personas get identical "citizen of {country}" prompt
prompts/opencharacter_persona.py — Demographic-weighted sampling from Census/WVS distributions per country
prompts/deep_persona.py        — 7 anchor attributes + taxonomy sampling (simplified DeepPersona)
engine/llm_client.py           — LocalLLM: vLLM offline inference, batch query_int_batch(), parse_int() (no server needed)
engine/run_survey.py           — Survey loop: load model -> batch inference per item (300/batch) -> save CSV (with resume)
engine/vllm_server.py          — Deprecated (was: vLLM server lifecycle)
metrics/step_a_gt_based.py     — WD, JSD, KS, MeanDiff per item, averaged (need GT distributions.json)
metrics/step_b_structural.py   — SFS = mean(SignF, SigF, NullF) (need both syn + real CSVs)
metrics/step_c_gt_free.py      — SCS, VCR, ICE computation functions (NO GT needed) — core contribution
metrics/step_c_compute_all.py  — Batch runner: computes Step C for all conditions, saves step_c_results.json
metrics/analysis.py            — Spearman/Kendall rank concordance: DI_combined vs WD/JSD/KS/MeanDiff
```

### Resume Support
`engine/run_survey.py` skips conditions where the output CSV already exists. Pipeline scripts use completion flags (`results/.0N_done`) to skip completed steps.

## Key Metrics

**GT-free (Step C — the core contribution)**:
- **SCS** (Synthetic Consistency Score): `DI_SCS = |alpha - 0.7|` — Cronbach's alpha deviation from healthy midpoint
- **VCR** (Value Coherence Ratio): `DI_VCR = max(0, lambda1/sum(lambda) - 0.5)` — first eigenvalue dominance (halo effect)
- **ICE** (Item Correlation Entropy): `DI_ICE = -H_norm(corr)` — negated normalized Shannon entropy of pairwise correlation distribution (20 bins, range [-1,1])
- **DI_combined**: min-max normalized average of DI_SCS, DI_VCR, DI_ICE (lower = better)

**GT-based (Step A)**: WD, JSD, KS, MeanDiff — per-item then averaged across 6 items

**Structural (Step B)**: SFS = mean(SignF, SigF, NullF) — correlation structure fidelity (needs both syn + real)

## Data

- **WVS Wave 7 GT**: `data/wvs_gt/distributions.json` — `{country: {question_code: {value_str: count}}}`
- **Privacy GT**: `data/privacy_gt/` and `data/privacy_data_gt/`
- **Raw privacy data**: `data/privacy_data/`
- **WVS preprocessing**: `data/preprocess_wvs.py`, `data/preprocess_privacy.py`
- **Synthetic results**: Auto-saved to `results/{domain}/{country}/{method}.csv`
- **Metrics output**: `results/metrics/step_a_results.json`, `step_c_results.json`, `analysis_results.json`

## Conventions

- All prompting modules expose: `generate_persona(country, n, seed=42, **kwargs) -> list[str]` returning system prompts
- Item columns in CSVs are prefixed `Q` (WVS) or `PC` (Privacy) — metrics modules auto-detect via `startswith(("Q", "PC"))`
- WVS items have varied scales (1-2, 1-3, 1-4, 1-5, 1-10); Privacy items are all 1-7 Likert
- LocalLLM uses vLLM offline mode (LLM class) — no server, GPU freed on process exit
- Step completion flags in `results/.0N_done`
- Pipeline uses `set -euo pipefail`; each step checks the previous step's completion flag
- All config is centralized in `config/experiment_config.py` — do not hardcode countries, items, or model settings elsewhere
- WVS countries: Argentina, Australia, Germany, India, Kenya, United States
- Privacy countries: South Africa, United Kingdom
