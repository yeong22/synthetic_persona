# Synthetic Persona Survey: GT-free Defect Indicators

LLM으로 생성한 합성(synthetic) 설문 데이터의 품질을 **Ground-Truth 없이** 평가하는 결함 지표(Defect Indicators) 연구.

## Research Question

> LLM synthetic persona의 설문 응답 품질을, 실제 인간 데이터(GT) 없이도 진단할 수 있는가?

## Metrics

| Step | Metric | GT 필요 | 설명 |
|------|--------|---------|------|
| A | WD, JSD, KS, Frobenius, α | O | 기존 분포 비교 메트릭 |
| B | SFS (SignF, SigF, NullF) | O | 상관 구조 충실도 |
| **C** | **SCS, VCR, ICE** | **X** | **핵심 — GT-free 결함 지표** |

- **SCS** (Synthetic Consistency Score): 내적 일관성(Cronbach's α)의 건강 범위 이탈 정도
- **VCR** (Value Coherence Ratio): PCA 제1고유값 지배도 → halo effect 탐지
- **ICE** (Item Correlation Entropy): 항목 간 상관 분포의 엔트로피 → 다양성 부족 탐지

## Setup

```bash
# Python 3.11+, NVIDIA GPU (RTX 3090 × 4 권장)
pip install -e .

# 또는 uv 사용
uv sync
```

## Quick Start

```bash
# 전체 파이프라인 한 번에 실행
bash run_all.sh
```

## Step-by-step

```bash
bash scripts/01_start_vllm.sh      # vLLM 서버 시작 (Qwen2.5-3B-Instruct)
bash scripts/02_run_domain_b.sh     # Domain B: WVS Wave 7 (6개국 × 3 prompting × 300 응답)
bash scripts/03_run_domain_a.sh     # Domain A: Privacy Calculus (2개국 × 3 prompting × 300 응답)
bash scripts/04_compute_metrics.sh  # 전체 메트릭 계산 (Step A + B + C)
bash scripts/05_analyze.sh          # DI ↔ GT-based 순위 일치도 분석
```

## Experiment Design

**모델**: Qwen/Qwen2.5-3B-Instruct (vLLM, tensor parallel 4)

**Prompting 전략**:
1. **Cultural Prompting** (Tao et al. 2024) — 국적 + 문화적 맥락만 제공
2. **OpenCharacter Persona** — 인구통계 속성 기반 캐릭터 프로필
3. **DeepPersona** — Big Five 성격 + 심층 사회경제적 프로필

**데이터**:
- Domain B: WVS Wave 7 — 6개국(KOR, USA, DEU, JPN, BRA, NGA) × 6 value items
- Domain A: Privacy Calculus — 2개국(KOR, USA) × 6 privacy items

## Project Structure

```
config/experiment_config.py      # 실험 설정
prompts/                         # 3가지 프롬프팅 전략
engine/                          # vLLM 클라이언트, 설문 시뮬레이션
metrics/
  step_a_gt_based.py             # WD, JSD, KS, Frobenius, α
  step_b_structural.py           # SFS (SignF, SigF, NullF)
  step_c_gt_free.py              # SCS, VCR, ICE ← 핵심
  analysis.py                    # DI↔GT 순위 일치 분석
scripts/01~05                    # 파이프라인 스크립트
results/                         # 실험 결과
```
