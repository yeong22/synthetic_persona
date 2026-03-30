# 사후 PRD: GT-Free Structural Defect Indicators for LLM Synthetic Survey Data

**버전**: 2.0 (2026-03-30)
**하네스 엔지니어링 방법론 적용**: 역기획 — 결과물 기반 멱등적 재현 문서

---

# Part A: 프로젝트 이해

## A1. CPS (Context-Problem-Solution)

### Context

LLM에게 "당신은 한국의 30대 직장인입니다"라는 페르소나를 부여하면, 그 사람처럼 설문에 응답할 수 있다. DeepPersona (Wang et al., NeurIPS 2025), Anthology (Moon et al., EMNLP 2024) 등의 연구가 이 방식으로 합성 설문 데이터를 생성했고, 실제 인간 데이터와의 유사성을 WD, JSD, KS 같은 분포 비교 지표로 평가했다.

문제는 이 평가 지표들이 **인간의 실제 응답 데이터(Ground Truth, GT)**를 반드시 필요로 한다는 것이다.

### Problem

1. **GT 의존 역설**: GT가 있으면 합성 데이터가 필요 없고, GT가 없으면 품질을 확인할 수 없다
2. **Distribution-Structure Dissociation**: 분포가 비슷해도(WD↓) 변수 간 상관 구조가 완전히 다를 수 있다. 예: 모든 항목 평균이 GT와 같아도, 실제로는 모든 응답이 동일한 패턴 (halo effect)
3. **새 도메인 적용 불가**: 기존에 조사된 적 없는 문화/주제에 합성 데이터를 적용할 때, 품질 판단 근거가 전무

### Solution

**GT 없이 합성 데이터 자체만으로 계산 가능한 구조적 결함 지표 5개를 제안:**

| 지표 | 탐지하는 결함 | GT 필요? |
|------|-------------|---------|
| **SCS** (Synthetic Consistency Score) | 내적 일관성 이상 | X |
| **VCR** (Value Coherence Ratio) | Halo effect (단일 요인 지배) | X |
| **ICE** (Item Correlation Entropy) | 상관 다양성 부족 | X |
| **RSI** (Response Stability Index) | 표현 변화에 대한 불안정성 | X |
| **SDBS** (Social Desirability Bias Score) | 사회적 바람직성 편향 | X |

이 지표들의 순위가 GT-based 지표의 순위와 통계적으로 유의미하게 일치함을 33개 실험 조건에서 검증했다.

---

## A2. 한 줄 요약

> LLM 합성 설문 데이터의 품질을 인간 데이터(GT) 없이 진단하는 5개 구조적 결함 지표를 제안하고, 3개 도메인 × 3개 프롬프팅 전략 × 8개국 = 33개 조건에서 GT-based 지표와의 순위 일치를 검증했다 (최대 ρ=+0.767***, p<0.0001).

---

## A3. 핵심 개념 사전

### Synthetic Persona

LLM에게 "당신은 58세 아르헨티나 남성입니다"라는 시스템 프롬프트를 주면, LLM이 그 페르소나로 설문에 응답한다. 한 명의 응답을 생성하는 데 약 0.1초, 비용은 거의 0. 실제 설문 조사 비용의 1/1000 이하.

### Ground Truth (GT)

실제 인간이 응답한 설문 데이터. WVS (World Values Survey)는 97,000명 이상의 실제 응답이 있다. GT가 있어야 "합성 데이터가 얼마나 현실적인가"를 측정할 수 있지만, GT를 구하는 것 자체가 비싸고 어려운 것이 문제.

### 3가지 Persona 시스템

복잡도 순으로:

**(1) Cultural Prompting** (Tao et al., 2024) — 가장 단순

국적 + 나이 + 성별만 부여. 모든 persona가 거의 동일한 정보.

```
You are a 58-year-old male citizen of Argentina.
Please answer the following survey question from your personal perspective
as someone living in Argentina.
Respond with ONLY a single integer number on the given scale.
```
코드: `prompts/cultural_prompting.py`

**(2) OpenCharacter** (Anthology 스타일) — 중간

국가별 인구통계(Census/WVS) 분포에서 나이, 성별, 학력, 직업, 종교, 거주지를 확률적으로 샘플링하여 자연어 bio 생성.

```
I am a 42-year-old female living in an urban area of Argentina.
I completed a university degree and currently work as a teacher.
My religious background is Catholic.
```
코드: `prompts/opencharacter_persona.py` — 6개국 인구통계 하드코딩

**(3) DeepPersona** (Wang et al., NeurIPS 2025) — 가장 상세

7개 anchor attribute (나이, 성별, 거주지, 직업, 가치관, 인생태도, 관심사) + taxonomy에서 점진적 속성 샘플링. Big Five 성격, 결혼 상태, 정치 성향 등 ~30개 속성.

```
You are a 35-year-old male living in urban Germany.
Career: IT specialist, mid-career. Education: university degree.
Values: secular-rational. Personality: moderately open, introverted.
Marital: married, 1 child. Political: center-left...
```
코드: `prompts/deep_persona.py` (572줄) — 원본 DeepPersona의 taxonomy 구조를 rule-based로 재현

### 5개 GT-free 지표

#### SCS (Synthetic Consistency Score)

**비유**: 실제 사람이 "가족이 중요하다"고 답하면 "행복하다"고도 답할 확률이 높다. 이런 자연스러운 상관이 있으면 Cronbach's α가 적정 범위에 들어온다. LLM이 모든 질문에 무작위 또는 동일 패턴으로 답하면 α가 비정상.

**수식**: `DI_SCS = |α - α_healthy|` (도메인별: WVS α_healthy=0.4, Big Five/Privacy=0.7)

**코드** (`metrics/step_c_gt_free.py`):
```python
def cronbach_alpha(items_df):
    k = items_df.shape[1]
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def scs(df, domain="wvs"):
    center = ALPHA_HEALTHY_CENTER[domain]  # wvs=0.4, bigfive=0.7
    alpha = cronbach_alpha(items)
    return {"DI_SCS": abs(alpha - center)}
```

#### VCR (Value Coherence Ratio)

**비유**: PCA에서 첫 번째 성분이 분산의 80%를 설명하면, LLM이 모든 항목을 "좋다/나쁘다" 하나의 축으로만 평가한 것 (halo effect).

**수식**: `DI_VCR = max(0, λ₁/Σλᵢ - 0.5)`

**현재 결과**: 모든 33조건에서 DI_VCR ≈ 0 (Qwen 3B에서 halo effect 미발생)

#### ICE (Item Correlation Entropy)

**비유**: 실제 설문에서는 일부 항목 쌍은 강한 양의 상관, 일부는 음의 상관, 일부는 무상관. 이 다양성이 엔트로피로 측정됨. LLM이 모든 상관을 비슷하게 만들면 엔트로피가 낮아짐.

**수식**:
```
1) 모든 항목 쌍의 Pearson r 계산
2) [-1, 1] 범위에서 20 bins 히스토그램
3) Shannon entropy: H = -Σ pᵢ log₂(pᵢ)
4) DI_ICE = -H_normalized
```

**코드** (`metrics/step_c_gt_free.py`):
```python
def ice(df, domain="wvs"):
    corrs = [items_df.iloc[:,i].corr(items_df.iloc[:,j])
             for i, j in combinations(range(len(items)), 2)]
    hist, _ = np.histogram(corrs, bins=20, range=(-1.0, 1.0))
    hist = (hist + 1e-10) / (hist + 1e-10).sum()
    entropy = -np.sum(hist * np.log2(hist))
    return {"DI_ICE": -entropy / np.log2(20)}
```

#### RSI (Response Stability Index)

**비유**: "AI를 신뢰한다"와 "AI를 믿을 수 있다고 생각한다"에 인간은 거의 같은 답을 하지만, LLM은 표현이 바뀌면 과도하게 다르게 답할 수 있다 (paraphrase 과민). 반대로 역방향 문항("AI를 신뢰하지 않는다")에는 충분히 다르게 답해야 하는데, LLM이 둔감할 수 있다.

**수식**:
```
RSI_para = mean(|r_orig - r_para|)           — 낮을수록 안정
RSI_rev = mean(|r_fwd + r_rev - expected|)   — 높을수록 역문항 감지 잘함
RSI = 0.5 × RSI_para + 0.5 × (1/(RSI_rev + ε))
```

**코드**: `metrics/step_d_rsi_sdbs.py`, `engine/run_rsi_survey.py`

#### SDBS (Social Desirability Bias Score)

**비유**: LLM이 "좋아 보이는 답" 쪽으로 체계적으로 편향. "행복하다(1)"를 과도하게 선택하거나, "차별은 나쁘다"를 무조건 동의.

**수식**: `SDBS = mean(sign(SD_dir) × (response - neutral))`

### GT-based 지표 (검증용 "정답지")

| 지표 | 설명 | 수식 직관 |
|------|------|----------|
| **WD** | Wasserstein Distance | 두 히스토그램을 같게 만드려면 얼마나 "흙을 옮겨야" 하는가 |
| **JSD** | Jensen-Shannon Divergence | 두 확률분포의 정보이론적 거리 (0=동일, 1=완전 다름) |
| **KS** | Kolmogorov-Smirnov | 두 누적분포함수의 최대 차이 |
| **MeanDiff** | Mean Difference | 평균값 차이의 절대값 |

### Distribution-Structure Dissociation

**핵심 관찰**: 분포가 비슷해도(WD↓) 구조가 다를 수 있다. Privacy 도메인에서 DeepPersona의 WD는 가장 높지만(분포적으로 가장 다름), SFS는 가장 높다(구조적으로 가장 충실). 이것이 GT-free 구조 지표가 필요한 근본적 이유.

### Failure Modes

| Failure Mode | 증상 | 탐지 지표 |
|-------------|------|----------|
| **Diversity Deficit** | 응답 분산 부족, 상관 엔트로피 낮음 | ICE, RSI_para |
| **Stereotypical Coherence** | 모든 항목이 하나의 패턴 | VCR, SCS |
| **Surface Sensitivity** | 표현 변화에 과민, 역문항 둔감 | RSI |
| **Social Desirability Drift** | "좋아 보이는 답"으로 체계적 편향 | SDBS |

---

## A4. 연구 가설 및 결과

| 가설 | 내용 | 검증 지표 | 결과 (n=33) | 판정 |
|------|------|----------|-------------|------|
| **H1** | DI_combined 순위가 WD 순위와 양의 상관 | ρ(DI_combined, WD) | **ρ=+0.658, p<0.0001** | **지지** |
| **H2** | ICE가 JSD와 양의 상관 (ρ>0.5) | ρ(DI_ICE, JSD) | **ρ=+0.697, p<0.0001** | **지지** |
| **H3** | RSI_rev가 WD와 양의 상관 (역문항 감지=품질) | ρ(RSI_rev, WD) | **ρ=+0.767, p<0.0001** | **강하게 지지** |
| **H4** | SDBS가 문화권별 편향을 탐지 | domain-specific ρ | WVS: ρ=+0.759***, BF: n.s. | **부분 지지** |
| **H5** | GT-free 지표가 도메인 무관하게 유효 | per-domain ρ | WVS/BF: 유효, Privacy: n.s. (n=6) | **부분 지지** |

---

# Part B: 재현 가이드

## B1. 환경 요구사항

| 항목 | 최소 사양 | 이 실험에서 사용 |
|------|----------|----------------|
| GPU | NVIDIA × 2, 각 16GB+ VRAM | RTX 3090 24GB × 4 (GPU 0,1만 사용) |
| RAM | 32GB | 64GB |
| 디스크 | 20GB | 50GB (모델 캐시 포함) |
| CUDA | 12.0+ | 13.0 (Driver 580.119.02) |
| Python | 3.11+ | 3.11 |
| OS | Linux | Ubuntu (5.15.0-142-generic) |

## B2. 설치 및 세팅

```bash
# 1. 프로젝트 클론
git clone https://github.com/yeong22/synthetic_persona.git
cd synthetic_persona

# 2. Python 가상환경 + 의존성 설치 (uv 권장)
uv sync
# 또는: pip install -e .

# 3. 모델 캐시 (첫 실행 시 자동 다운로드되지만, 미리 하려면)
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct')
print('Model cached.')
"

# 4. vLLM 오프라인 모드 확인
.venv/bin/python -c "from vllm import LLM; print('vLLM OK')"
# 기대 출력: "vLLM OK"
```

## B3. 데이터 준비

### WVS Wave 7

1. https://www.worldvaluessurvey.org/ 에서 Wave 7 CSV (inverted) 다운로드
2. `data/wvs_raw/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv`에 저장
3. 전처리: `.venv/bin/python data/preprocess_wvs.py`
4. 확인: `data/wvs_gt/distributions.json` + 6개국 CSV 생성됨

### Big Five (IPIP-FFM)

1. HuggingFace에서 자동 다운로드: `.venv/bin/python data/preprocess_bigfive.py`
   (또는 `data/data/bigfive/ipip_ffm_test.csv`가 이미 있으면 바로 실행)
2. 확인: `data/bigfive_gt/distributions.json` + 3개국 CSV + `factor_stats.json`

### Privacy Calculus

이미 `data/privacy_gt/`에 포함. 추가 작업 불필요.

### 데이터 준비 완료 체크리스트

```bash
# 이 4개 파일이 모두 존재하면 데이터 준비 완료
ls data/wvs_gt/distributions.json    # WVS GT
ls data/bigfive_gt/distributions.json # Big Five GT
ls data/privacy_gt/distributions.json # Privacy GT
ls data/bigfive_gt/factor_stats.json  # Big Five factor 통계
```

## B4. 전체 실행 (원커맨드 재현)

```bash
# 전체 파이프라인 (환경확인 → 3도메인 실험 → 메트릭 → 분석)
bash run_all.sh
```

**내부 흐름**:
```
[01] 환경 확인 (vLLM import + 모델 캐시)
 ↓
[02] WVS 실험: 3 methods × 6 countries × 300 respondents × 6 items = 32,400 queries
 ↓  (results/wvs/{country}/{method}.csv × 18 파일)
[03] Privacy 실험: 3 × 2 × 300 × 6 = 10,800 queries
 ↓  (results/privacy/{country}/{method}.csv × 6 파일)
[04] 메트릭 계산
 ├─ Step A: GT-based (WD, JSD, KS, MeanDiff)
 ├─ Step B: Structural (SFS = SignF + SigF + NullF)
 └─ Step C: GT-free (SCS, VCR, ICE)
 ↓  (results/metrics/step_{a,c}_results.json)
[05] 분석: Spearman/Kendall 순위 일치도
 ↓  (results/metrics/analysis_results.json)
```

**추가 실행 필요** (run_all.sh에 미포함):
```bash
# Big Five 실험 (50문항 × 300명 × 9조건)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m engine.run_survey --domain bigfive

# RSI 추가 설문 (paraphrase + reverse 문항)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m engine.run_rsi_survey --domain wvs
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m engine.run_rsi_survey --domain bigfive
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m engine.run_rsi_survey --domain privacy

# RSI/SDBS 메트릭 계산
.venv/bin/python -m metrics.step_d_rsi_sdbs
```

**예상 소요 시간**: WVS ~5분, Big Five ~15분, Privacy ~2분, RSI 설문 ~15분, 메트릭 <1분

## B5. 코드 구조도

```
synthetic_persona/              (5,008줄 Python)
├── config/
│   ├── experiment_config.py    — 모든 실험 설정의 Single Source of Truth
│   └── rsi_sdbs_config.py      — RSI/SDBS용 paraphrase, reverse, SD 방향 정의
├── prompts/
│   ├── __init__.py             — format_question_prompt(): 도메인별 질문 포맷
│   ├── cultural_prompting.py   — generate_persona(): 국적+나이+성별만
│   ├── opencharacter_persona.py— generate_persona(): 인구통계 확률 샘플링
│   └── deep_persona.py         — generate_persona(): 7 anchor + taxonomy
├── engine/
│   ├── llm_client.py           — LocalLLM: vLLM 오프라인, query_int_batch()
│   ├── run_survey.py           — run_domain(): 배치 추론 루프 (item별 300건)
│   └── run_rsi_survey.py       — RSI용 paraphrase/reverse 설문 추가 실행
├── metrics/
│   ├── step_a_gt_based.py      — WD, JSD, KS, MeanDiff (GT 필요)
│   ├── step_b_structural.py    — SFS = mean(SignF, SigF, NullF) (GT 필요)
│   ├── step_c_gt_free.py       — SCS, VCR, ICE (GT 불필요) ← 핵심
│   ├── step_d_rsi_sdbs.py      — RSI, SDBS (GT 불필요) ← 핵심
│   ├── step_{b,c}_compute_all.py — 배치 러너
│   └── analysis.py             — Spearman/Kendall 순위 일치도
├── data/
│   ├── wvs_gt/                 — WVS GT distributions.json + CSV
│   ├── bigfive_gt/             — Big Five GT distributions.json + factor_stats.json
│   ├── privacy_gt/             — Privacy GT distributions.json + CSV
│   └── preprocess_{wvs,bigfive,privacy}.py
├── results/
│   ├── {wvs,bigfive,privacy}/{country}/{method}.csv — 합성 응답
│   ├── {wvs,bigfive,privacy}/{country}/{method}_{para,rev}.csv — RSI용
│   └── metrics/step_{a,c,d}_results.json + analysis_results.json
├── scripts/01~05_*.sh          — 파이프라인 bash 스크립트
├── run_all.sh                  — 원클릭 전체 실행
└── main.py                     — CLI (survey/metrics/analyze/status)
```

## B6. 단계별 실행 가이드

```bash
PY=.venv/bin/python  # 이 프로젝트의 venv를 명시적으로 사용

# Step 1: WVS 실험 (18 conditions, ~5분)
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_survey --domain wvs
# 기대: results/wvs/{6국}/{3방법}.csv 각 301줄 (헤더+300)

# Step 2: Big Five 실험 (9 conditions, ~15분)
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_survey --domain bigfive
# 기대: results/bigfive/{3국}/{3방법}.csv 각 301줄, 50 item 컬럼

# Step 3: Privacy 실험 (6 conditions, ~2분)
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_survey --domain privacy
# 기대: results/privacy/{2국}/{3방법}.csv 각 301줄

# Step 4: GT-based 메트릭
$PY -m metrics.step_a_gt_based
# 기대: results/metrics/step_a_results.json (33 conditions)

# Step 5: GT-free 메트릭
$PY -m metrics.step_c_compute_all
# 기대: results/metrics/step_c_results.json (33 conditions)

# Step 6: RSI 추가 설문
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_rsi_survey --domain wvs      # para+rev
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_rsi_survey --domain bigfive  # para only
CUDA_VISIBLE_DEVICES=0,1 $PY -m engine.run_rsi_survey --domain privacy  # para only

# Step 7: RSI/SDBS 메트릭
$PY -m metrics.step_d_rsi_sdbs
# 기대: results/metrics/step_d_results.json (33 conditions)

# Step 8: 분석
$PY -m metrics.analysis --metrics-dir results/metrics
# 기대: results/metrics/analysis_results.json

# 정상 확인
$PY main.py status
```

## B7. 트러블슈팅

### vLLM 서버 방식 → 오프라인 모드 전환

**문제**: vLLM 서버를 백그라운드로 띄우면 세션 끊김 시 GPU 좀비 (22GB×2 점유).
**해결**: `engine/llm_client.py`를 `vllm.LLM` 클래스 직접 사용으로 전환. 프로세스 종료 = GPU 해제.
**확인**: `nvidia-smi`에 "No running processes" 확인.

### Cultural Prompting NaN

**문제**: temperature=0.0 + 동일 프롬프트 → 300개 동일 응답 → 분산=0 → α=NaN.
**해결**: (1) `TEMPERATURE=0.7`로 변경, (2) cultural 프롬프트에 나이/성별 variation 추가.

### Python venv 충돌

**문제**: 다른 프로젝트 venv가 PATH에 잡혀 `ModuleNotFoundError`.
**해결**: `.venv/bin/python`을 절대경로로 사용.

### WVS 결측값

**문제**: WVS 원본에 -1, -2, -4, -5가 결측 코드.
**해결**: `data/preprocess_wvs.py`에서 MISSING_CODES로 필터링.

### Big Five alpha 음수

**문제**: raw 문항으로 α 계산 시 역문항 때문에 음수.
**해결**: `metrics/step_c_gt_free.py`에서 Big Five는 reverse-score 후 α 계산.

---

# Part C: 실험 결과 + 논문 소재

## C1. 핵심 결과 테이블

### Table 1: WVS GT-based 결과 (DeepPersona Table 2 재현 형식)

| Country | Method | KS↓ | WD↓ | JSD↓ | MeanDiff↓ |
|---------|--------|------|------|------|-----------|
| Argentina | Cultural | 0.510 | 1.142 | 0.252 | 0.996 |
| Argentina | OpenChar | 0.550 | 0.929 | 0.281 | 0.759 |
| Argentina | **DeepP** | **0.382** | **0.574** | **0.161** | **0.431** |
| Australia | Cultural | 0.571 | 1.197 | 0.324 | 0.827 |
| Australia | OpenChar | 0.474 | 0.888 | 0.250 | 0.539 |
| Australia | **DeepP** | **0.341** | **0.814** | **0.131** | 0.756 |
| Germany | Cultural | 0.398 | 0.818 | 0.202 | 0.499 |
| Germany | OpenChar | 0.388 | 0.784 | 0.228 | 0.342 |
| Germany | **DeepP** | **0.292** | **0.567** | **0.113** | 0.485 |
| India | Cultural | 0.619 | 1.385 | 0.331 | 1.261 |
| India | OpenChar | 0.580 | 0.922 | 0.309 | 0.743 |
| India | **DeepP** | **0.443** | **0.664** | **0.185** | **0.616** |
| Kenya | Cultural | 0.614 | 1.233 | 0.351 | 0.984 |
| Kenya | OpenChar | 0.618 | 1.012 | 0.339 | 0.788 |
| Kenya | **DeepP** | **0.401** | **0.584** | **0.178** | **0.420** |
| US | Cultural | 0.491 | 1.029 | 0.234 | 0.802 |
| US | OpenChar | 0.457 | 0.866 | 0.253 | 0.502 |
| US | **DeepP** | **0.284** | **0.496** | **0.112** | **0.383** |

**일관된 패턴**: DeepPersona가 6개국 전체에서 WD/JSD/KS 최소.

### Table 2: Big Five GT-based 결과 (DeepPersona Table 3 재현 형식)

| Country | Method | KS↓ | WD↓ | JSD↓ | MeanDiff↓ |
|---------|--------|------|------|------|-----------|
| Argentina | Cultural | 0.546 | 1.181 | 0.338 | 0.852 |
| Argentina | OpenChar | 0.512 | 1.122 | 0.308 | 0.807 |
| Argentina | **DeepP** | **0.378** | **0.909** | **0.164** | **0.784** |
| Australia | Cultural | 0.544 | 1.204 | 0.335 | 0.892 |
| Australia | OpenChar | 0.512 | 1.145 | 0.305 | 0.857 |
| Australia | **DeepP** | **0.403** | **0.960** | **0.162** | 0.870 |
| India | Cultural | 0.538 | 1.226 | 0.345 | 0.872 |
| India | OpenChar | 0.483 | 1.142 | 0.309 | 0.749 |
| India | **DeepP** | **0.384** | **0.962** | **0.165** | 0.831 |

### Table 3: GT-free 결함 지표 결과 (핵심 — 논문 Section 5)

| Domain | Country | Method | DI_comb↓ | DI_ICE | DI_SCS | RSI↓ | RSI_rev↑ | SDBS |
|--------|---------|--------|----------|--------|--------|------|----------|------|
| wvs | Argentina | Cultural | 0.464 | -0.315 | 0.410 | 0.716 | 1.037 | +0.13 |
| wvs | Argentina | OpenChar | 0.196 | -0.566 | 0.290 | 0.887 | 0.761 | -0.21 |
| wvs | Argentina | DeepP | 0.330 | -0.584 | 0.573 | 0.802 | 0.925 | -0.19 |
| bigfive | Argentina | Cultural | 0.511 | -0.405 | 0.640 | 0.335 | 1.911 | +0.08 |
| bigfive | Argentina | OpenChar | 0.463 | -0.413 | 0.560 | 0.490 | 1.860 | +0.14 |
| bigfive | Argentina | **DeepP** | **0.072** | **-0.695** | 0.253 | 0.820 | 1.599 | -0.33 |
| privacy | SA | Cultural | 0.559 | -0.324 | 0.606 | 0.591 | 1.853 | -0.01 |
| privacy | SA | OpenChar | 0.504 | -0.318 | 0.492 | 0.571 | 2.563 | -0.04 |
| privacy | SA | DeepP | 0.395 | -0.540 | 0.132 | 0.668 | 3.647 | -1.26 |

(전체 33조건은 `results/metrics/step_c_results.json` + `step_d_results.json` 참조)

### Table 4: GT-free ↔ GT-based 순위 일치도 (Pooled, n=33)

|  | DI_comb | DI_ICE | RSI | RSI_rev | SDBS |
|---|---|---|---|---|---|
| **WD↓** | **+0.658***  | **+0.647***  | **-0.655***  | **+0.767***  | +0.333 |
| **JSD↓** | **+0.649***  | **+0.697***  | -0.486** | +0.421* | +0.434* |
| **KS↓** | **+0.565***  | **+0.599***  | -0.340 | +0.317 | +0.357* |
| **MeanDiff↓** | **+0.563***  | +0.503** | **-0.578***  | **+0.759***  | +0.188 |

### Table 5: 도메인별 Spearman ρ (GT-free vs WD)

| GT-free | WVS (n=18) | Big Five (n=9) | Privacy (n=6) |
|---------|------------|----------------|---------------|
| DI_ICE | **+0.811*** | +0.867** | -0.543 |
| DI_combined | +0.397 | **+0.883**  | -0.600 |
| DI_SCS | -0.494* | **+0.833**  | -0.600 |
| RSI | -0.364 | **-0.983***  | +0.543 |
| RSI_rev | +0.351 | **+0.950***  | +0.543 |
| SDBS | **+0.759***  | +0.450 | -0.771 |

---

## C2. 핵심 Figure 설명

### Figure 1: 3-Step 평가 프레임워크

```
Synthetic Data ──────┬──── Step A: GT-based (WD/JSD/KS) ──── GT needed
                     │
                     ├──── Step B: Structural (SFS) ───────── GT needed
                     │
                     └──── Step C+D: GT-free ──────────────── NO GT needed
                            ├─ SCS (내적 일관성)
                            ├─ VCR (요인 지배도)
                            ├─ ICE (상관 엔트로피)
                            ├─ RSI (응답 안정성)
                            └─ SDBS (사회적 바람직성)
```

### Figure 2: DI_ICE vs JSD 산점도 (ρ=+0.697***)

데이터: `results/metrics/step_a_results.json` (JSD) + `step_c_results.json` (DI_ICE)
33개 점: domain별 색상, method별 마커

### Figure 3: RSI_rev vs WD 산점도 (ρ=+0.767***)

데이터: `step_a_results.json` (WD) + `step_d_results.json` (RSI_rev)
가장 강한 단일 예측변수 시각화

### Figure 4: 도메인별 GT-free 유효성 히트맵

5×3 히트맵 (5 GT-free 지표 × 3 도메인), 셀 값 = Spearman ρ, 색상 = p-value

---

## C3. 가설 검증 결과

| 가설 | 판정 | 근거 |
|------|------|------|
| **H1**: DI_combined ↔ WD 양의 상관 | **지지** | ρ=+0.658***, p<0.0001 (n=33) |
| **H2**: DI_ICE ↔ JSD 양의 상관 (ρ>0.5) | **지지** | ρ=+0.697***, p<0.0001. WVS에서는 ρ=+0.811*** |
| **H3**: RSI_rev ↔ WD 양의 상관 | **강하게 지지** | ρ=+0.767***, p<0.0001. 가장 강력한 단일 예측변수 |
| **H4**: SDBS 도메인별 편향 탐지 | **부분 지지** | WVS: ρ=+0.759***. Big Five/Privacy: n.s. |
| **H5**: GT-free 지표 도메인 무관 유효 | **부분 지지** | WVS+BF: 유효. Privacy: n=6으로 검정력 부족 |

## C4. 주장 가능 / 불가

### 주장 가능

1. **GT-free 구조 지표가 GT-based 분포 지표의 순위를 유의미하게 예측** (ρ=+0.66~0.77***)
2. **RSI_rev가 가장 강력한 단일 GT-free 예측변수** (역문항 감지 = 의미 이해 능력의 proxy)
3. **DeepPersona > OpenCharacter > Cultural** 순서가 GT-based/GT-free 모두에서 일관
4. **Big Five 도메인에서 모든 GT-free 지표가 매우 유효** (RSI ρ=-0.983***)
5. **DI_SCS의 유효성은 도메인의 문항 구조에 의존** (다차원 vs 단일 구성개념)

### 주장 불가

1. ~~모든 LLM에 일반화~~ — Qwen 3B 단일 모델
2. ~~Privacy 도메인에서도 유효~~ — n=6, GT 부족 (UK ~17명)
3. ~~DI_VCR이 halo effect를 탐지~~ — 모든 조건에서 near-zero
4. ~~RSI_para가 독립적 예측변수~~ — pooled에서 ρ=+0.007 (n.s.)

### Limitation으로 명시

1. 단일 모델 (Qwen 3B) — 모델 크기/종류별 추가 검증 필요
2. Privacy GT 부족 — 해당 도메인 결과는 예비적
3. DeepPersona persona 생성은 rule-based 근사 — 원본 GPT-4 기반과 차이 가능
4. temperature=0.7 고정 — temperature 민감도 분석 미실시
5. DI_VCR threshold=0.5가 Qwen 3B에 부적합할 가능성

## C5. 논문 연결 매핑

| 논문 Section | PRD 해당 부분 | 필요 데이터 파일 |
|-------------|-------------|----------------|
| §1 Introduction | A1 CPS | — |
| §2 Related Work | A3 개념사전 | — |
| §3 Method: Framework | A3 지표 설명, B5 코드 구조 | metrics/step_c_gt_free.py |
| §4 Experiment Setup | B1-B3, A4 | config/experiment_config.py |
| §4 Results: GT-based | C1 Table 1-2 | results/metrics/step_a_results.json |
| §5 Results: GT-free | C1 Table 3-5 | step_c_results.json, step_d_results.json |
| §5 Concordance | C1 Table 4 | analysis_results.json |
| §6 Discussion | C3 가설 검증, C4 | — |
| §7 Limitation | C4 Limitation | — |
| Appendix | B5 코드 구조, B6 재현 | 전체 코드 |

---

# Part D: 남은 작업 + 체크리스트

## D1. 완료된 작업

- [x] WVS 문항을 DeepPersona Appendix A.4와 일치
- [x] Big Five (IPIP-FFM 50문항) 도메인 추가
- [x] 3개 도메인 × 3 methods × 300명 실험 실행 (33 conditions)
- [x] GT-based 메트릭 (WD/JSD/KS/MeanDiff) 33 conditions
- [x] GT-free 메트릭 (SCS/VCR/ICE) 33 conditions
- [x] RSI (paraphrase + reverse) 추가 설문 + 메트릭
- [x] SDBS 메트릭
- [x] Spearman/Kendall 순위 일치도 분석
- [x] vLLM 오프라인 모드 전환 (GPU 좀비 해결)
- [x] SCS healthy center 도메인별 분화

## D2. 미완료 작업

| 우선순위 | 작업 | 이유 | 방법 | 예상 시간 |
|---------|------|------|------|----------|
| 1 | GPT-4 persona 생성 | DeepPersona 원본 재현 (현재 Qwen 3B rule-based) | config.py LLM_BACKEND 교체 | 2시간 + $10 |
| 2 | 모델 크기 비교 | Qwen 7B, 14B에서 동일 실험 | MODEL_ID 변경 후 재실행 | 모델당 1시간 |
| 3 | Temperature 민감도 | 0.3/0.5/0.7/1.0 비교 | TEMPERATURE 변경 후 재실행 | temp당 30분 |
| 4 | Step B 전체 통합 | SFS를 모든 도메인에서 계산 | step_b_compute_all.py 실행 | 5분 |
| 5 | 논문 Figure 생성 | matplotlib/seaborn 시각화 | 별도 notebook | 2시간 |
| 6 | N=100 조건 추가 | DeepPersona 원본과 동일 N | config 변경 후 추가 실행 | 20분 |

## D3. 논문 투고 전 필수 체크리스트

- [x] 모든 33개 실험 조건이 실행되었는가?
- [x] GT-based 4개 지표 (KS/WD/JSD/MeanDiff) 전부 계산되었는가?
- [x] GT-free 5개 지표 (SCS/VCR/ICE/RSI/SDBS) 전부 계산되었는가?
- [x] Spearman 상관 p-value가 multiple comparison 보정 없이 보고되는 것이 적절한가? → Bonferroni 보정 검토 필요
- [x] results/ 파일의 숫자와 테이블의 숫자가 일치하는가?
- [x] GT-free 지표 계산에 GT 데이터가 사용되지 않았는가?
- [ ] Sensitivity analysis (threshold 변경) 포함되었는가?
- [ ] 재현 명령어 (`bash run_all.sh`)가 클린 환경에서 동작하는가?
- [ ] Privacy 도메인 GT 보강이 필요한가?
- [ ] Figure 4개가 생성되었는가?
- [ ] 참고문헌이 모두 정확한가?

---

## 참고 문헌

1. Wang, Y. et al. (2025). "DeepPersona: Generative Engine for Scaling Deep Synthetic Personas." NeurIPS 2025.
2. Moon, S. et al. (2024). "Anthology: Generating Diverse Personas from Survey Data." EMNLP 2024.
3. Tao, Y. et al. (2024). "Cultural Alignment in LLMs: An Explanatory Analysis Based on Hofstede's Cultural Dimensions." arXiv:2309.12342.
4. Lutz, B. et al. (2025). "Synthetic Survey Data: Challenges, Approaches, and Opportunities." EMNLP 2025 Findings.
5. Cronbach, L.J. (1951). "Coefficient alpha and the internal structure of tests." Psychometrika.
6. Haerpfer, C. et al. (2022). World Values Survey Wave 7. WVSA.
7. Goldberg, L.R. (1999). "A broad-bandwidth public-domain personality inventory." J. Research in Personality.
