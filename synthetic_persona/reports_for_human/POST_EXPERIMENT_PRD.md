# 사후 PRD: GT-Free Structural Defect Indicators for LLM Synthetic Survey Data

---

## 1. 프로젝트 개요

### 한 줄 요약

LLM이 생성한 합성(synthetic) 설문 데이터의 품질을, 실제 인간 데이터(Ground Truth) 없이 진단할 수 있는 3가지 구조적 결함 지표(SCS, VCR, ICE)를 제안하고 검증하는 연구.

### 연구 배경: 왜 이 연구가 필요한가

최근 사회과학에서 LLM을 "합성 설문 응답자"로 사용하는 연구가 급증하고 있다. "한국 30대 직장인"이라는 페르소나를 LLM에 부여하면, 그 페르소나가 실제 설문에 어떻게 응답할지를 시뮬레이션할 수 있다. 이런 합성 데이터는 실제 설문 비용의 1% 미만으로 수천 건의 응답을 생성할 수 있어 매력적이다.

**문제는 품질 검증이다.** 합성 데이터가 실제 인간 데이터와 얼마나 비슷한지를 검증하려면, 비교 대상인 실제 인간 데이터(Ground Truth, GT)가 필요하다. 그런데:

1. GT가 있으면 애초에 합성 데이터가 필요 없다 (GT를 직접 쓰면 되니까)
2. GT가 없는 새로운 문화/맥락에 합성 데이터를 적용할 때, 품질을 검증할 방법이 없다
3. 기존 GT-based 메트릭(WD, JSD 등)은 GT가 반드시 있어야 계산 가능하다

**이 연구의 핵심 질문:** GT 없이, 합성 데이터의 내부 구조만 보고 "이 데이터는 결함이 있다"고 진단할 수 있는가?

### 핵심 가설

> **H1**: GT-free 결함 지표(DI_combined = SCS + VCR + ICE의 정규화 평균)의 조건별 순위가, GT-based 메트릭(WD, JSD)의 순위와 통계적으로 유의미하게 일치한다.

구체적으로:
- **H1a**: DI_ICE(Item Correlation Entropy)가 JSD와 양의 상관 (ρ > 0.5, p < 0.05)
- **H1b**: DI_combined가 JSD와 양의 상관 (ρ > 0.5, p < 0.05)
- **H1c**: Per-country 순위 일치율이 우연(33%) 이상

### 타겟 학회

EMNLP 2026 Workshop (예: SyntheticData Workshop, SoLLM)

---

## 2. 핵심 개념 설명

### 2.1. 세 가지 Persona 프롬프팅 시스템

LLM에게 "당신은 ○○입니다"라는 시스템 프롬프트(persona)를 부여한 뒤, 설문 문항에 응답하게 한다. persona의 상세도에 따라 3가지 전략을 비교한다.

#### (1) Cultural Prompting (Tao et al. 2024) — 가장 단순

국적 + 최소한의 인구통계(나이, 성별)만 부여. 모든 persona가 거의 동일한 정보를 가짐.

**실제 프롬프트 예시** (`prompts/cultural_prompting.py`에서 발췌):
```
You are a 58-year-old male citizen of Argentina. Please answer the following
survey question from your personal perspective as someone living in Argentina.
Respond with ONLY a single integer number on the given scale.
Do not add any explanation or reasoning.
```

```
You are a 19-year-old female citizen of Argentina. Please answer the following
survey question from your personal perspective as someone living in Argentina.
Respond with ONLY a single integer number on the given scale.
Do not add any explanation or reasoning.
```

**코드** (`prompts/cultural_prompting.py`):
```python
def generate_persona(country: str, n: int, seed: int = 42, **kwargs) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        age = rng.randint(18, 80)
        gender = rng.choice(["male", "female"])
        prompts.append(
            f"You are a {age}-year-old {gender} citizen of {country}. "
            f"Please answer the following survey question from your personal perspective "
            f"as someone living in {country}. "
            f"Respond with ONLY a single integer number on the given scale. "
            f"Do not add any explanation or reasoning."
        )
    return prompts
```

#### (2) OpenCharacter Persona — 중간 수준

국가별 인구통계 분포(Census/WVS 기반)에서 나이, 성별, 학력, 직업, 종교, 거주지역을 확률적으로 샘플링하여 3-5문장짜리 자연어 bio를 생성.

**실제 프롬프트 예시** (생성된 bio 일부):
```
You are Maria, a 42-year-old female living in urban Argentina. You completed
a university degree and work as a teacher. You identify as Catholic. You live
in a bustling city neighborhood. Please answer the following survey question
from your personal perspective...
```

**핵심 차이**: Cultural은 국적+나이+성별만 제공하지만, OpenCharacter는 학력, 직업, 종교, 거주지역까지 구체적으로 부여한다. 각 속성은 해당 국가의 실제 인구통계 비율에 따라 확률적으로 샘플링된다 (예: 아르헨티나 Catholic 55%, Evangelical 15%, non-religious 25%).

#### (3) DeepPersona (Wang et al., NeurIPS 2025) — 가장 상세

7개 anchor attribute(나이, 성별, 거주지, 직업, 가치관, 인생 태도, 관심사) + 200+ taxonomy attribute에서 점진적으로 세부 속성을 샘플링. Big Five 성격 특성, 사회경제적 상세 프로필까지 포함.

**실제 프롬프트 예시** (생성된 persona 일부):
```
You are a 35-year-old male living in urban Germany.
Career: IT specialist, mid-career, employed full-time
Education: university degree (Informatik)
Values: secular-rational — you prioritize logic and evidence over tradition
Personality: moderately open, moderately conscientious, introverted
Life attitude: pragmatic
Marital status: married, 1 child
Hobbies: hiking, board games
Political leaning: center-left
Media consumption: primarily online news
Financial situation: comfortable, owns apartment...
```

**핵심 차이**: anchor 값에 따라 세부 속성이 일관되게 결정된다 (예: `young_adult` → `single` 선호, `senior` → `married/widowed` 선호). 이 일관성이 더 현실적인 persona를 만든다.

### 2.2. 세 가지 GT-free 결함 지표 (핵심 기여)

이 지표들은 합성 데이터의 **내부 구조**만 분석하여 결함을 탐지한다. GT가 전혀 필요 없다.

#### (1) SCS (Synthetic Consistency Score) — "내적 일관성 진단"

**측정하는 것**: 설문 항목 간 내적 일관성(Cronbach's α)이 "건강한 범위"에서 얼마나 벗어났는가.

**비유**: 실제 사람이 "가족이 중요하다"고 답하면 "친구도 중요하다"고 답할 확률이 높다. 이런 자연스러운 상관이 있으면 α가 0.5-0.9 범위에 들어온다. LLM이 모든 질문에 동일한 패턴으로 답하면 α가 비정상적으로 높거나 낮아진다.

**수식**:
```
Cronbach's α = (k / (k-1)) × (1 - Σ(item_variances) / total_variance)
DI_SCS = |α - 0.7|    (0.7 = 건강 범위 중심점; 낮을수록 좋음)
```

**실제 코드** (`metrics/step_c_gt_free.py`):
```python
ALPHA_HEALTHY_CENTER = 0.7

def cronbach_alpha(df):
    items = df[_item_columns(df)].dropna()
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def scs(df):
    alpha = cronbach_alpha(df)
    di_scs = abs(alpha - ALPHA_HEALTHY_CENTER)
    return {"alpha": alpha, "DI_SCS": di_scs}
```

**실제 결과 예시**:
- Argentina/cultural: α=0.024 → DI_SCS=0.676 (α가 너무 낮음 → 결함)
- Argentina/deep_persona: α=-0.207 → DI_SCS=0.907 (음의 α → 역상관 패턴)
- South Africa/deep_persona: α=0.627 → DI_SCS=0.073 (건강 범위에 근접)

#### (2) VCR (Value Coherence Ratio) — "Halo Effect 탐지"

**측정하는 것**: PCA의 첫 번째 고유값이 전체 분산에서 차지하는 비율. LLM이 모든 항목을 하나의 차원(예: "좋다/나쁘다")으로만 답하면 이 비율이 매우 높아진다.

**비유**: 실제 사람에게 "가족 중요도", "신뢰", "삶의 만족" 등을 물으면 각각 다른 차원의 답변이 나온다. 하지만 LLM이 "이 사람은 긍정적이니 모든 질문에 높은 점수를 줘야지"라고 추론하면, 모든 항목이 하나의 요인으로 수렴한다 (halo effect).

**수식**:
```
VCR = λ₁ / Σλᵢ    (첫 번째 고유값 / 전체 고유값 합)
DI_VCR = max(0, VCR - 0.5)    (VCR > 0.5일 때만 결함; 낮을수록 좋음)
```

**실제 코드** (`metrics/step_c_gt_free.py`):
```python
VCR_THRESHOLD = 0.5

def vcr(df):
    items = df[_item_columns(df)].dropna()
    items = items.loc[:, items.var(ddof=1) > 0]  # zero-variance 항목 제외
    corr_matrix = items.corr().values
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    vcr_val = eigenvalues[0] / eigenvalues.sum()
    di_vcr = max(0.0, vcr_val - VCR_THRESHOLD)
    return {"VCR": vcr_val, "DI_VCR": di_vcr}
```

**실제 결과**: 모든 24개 조건에서 DI_VCR = 0.0. Qwen2.5-3B 모델은 halo effect를 생성하지 않는 것으로 보임. → 이 지표는 현재 실험에서는 변별력 없음.

#### (3) ICE (Item Correlation Entropy) — "상관 다양성 진단" ⭐ 가장 유망

**측정하는 것**: 항목 간 pairwise 상관계수 분포의 Shannon 엔트로피. 상관계수들이 다양하면 엔트로피가 높고, 한쪽에 몰려 있으면 낮다.

**비유**: 실제 설문에서는 일부 항목은 강하게 상관되고(예: 가족 vs 친구), 일부는 약하거나 음의 상관을 보인다(예: 신뢰 vs 보수성). 이 다양성이 자연스러운 것이다. LLM이 모든 상관을 비슷하게 만들면 엔트로피가 낮아진다.

**수식**:
```
1) 모든 항목 쌍의 Pearson 상관계수 계산 → corrs = [r₁₂, r₁₃, ..., r₅₆]
2) [-1, 1] 범위에서 20 bins의 히스토그램 생성
3) 히스토그램을 확률 분포로 정규화
4) Shannon entropy: H = -Σ pᵢ log₂(pᵢ)
5) 정규화: H_norm = H / log₂(20)
6) DI_ICE = -H_norm    (엔트로피가 높을수록 DI가 낮음 = 좋음)
```

**실제 코드** (`metrics/step_c_gt_free.py`):
```python
def ice(df):
    items_df = df[_item_columns(df)].dropna()
    items = items_df.columns.tolist()

    corrs = []
    for i, j in combinations(range(len(items)), 2):
        r = items_df.iloc[:, i].corr(items_df.iloc[:, j])
        if not np.isnan(r):
            corrs.append(r)

    hist, _ = np.histogram(corrs, bins=20, range=(-1.0, 1.0), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    max_entropy = np.log2(len(hist))
    normalized_entropy = entropy / max_entropy

    return {
        "ICE": float(entropy),
        "ICE_normalized": float(normalized_entropy),
        "DI_ICE": float(-normalized_entropy),
    }
```

**실제 결과 예시**:
- Argentina/cultural: ICE_norm=0.444, DI_ICE=-0.444 (상관이 한쪽에 치우침 → 나쁨)
- Argentina/deep_persona: ICE_norm=0.630, DI_ICE=-0.630 (상관이 다양 → 좋음)

**왜 ICE가 가장 유망한가**: 이 실험에서 ICE는 JSD와 ρ=+0.797 (p<0.001)의 매우 강한 양의 상관을 보였다. GT 없이도 합성 데이터의 품질 순위를 높은 정확도로 예측할 수 있다.

### 2.3. GT-based 지표 (검증용)

이 지표들은 합성 데이터와 실제 인간 데이터(GT)의 분포를 직접 비교한다. GT-free 지표의 유효성을 검증하는 "정답지" 역할.

| 지표 | 설명 | 계산 | 해석 |
|------|------|------|------|
| **WD** | Wasserstein Distance | 두 분포 간 "흙 옮기기" 최소 비용 | 낮을수록 유사 |
| **JSD** | Jensen-Shannon Divergence | 두 분포의 정보이론적 거리 (0~1 범위, 대칭) | 낮을수록 유사 |
| **KS** | Kolmogorov-Smirnov | 두 CDF의 최대 수직 거리 | 낮을수록 유사 |
| **MeanDiff** | Mean Difference | |mean(syn) - mean(GT)| | 낮을수록 유사 |

각 지표는 6개 항목별로 계산된 뒤 평균을 취한다.

### 2.4. Distribution-Structure Dissociation

이 연구의 숨은 관찰: **분포적 유사성(WD/JSD)과 구조적 유사성(SFS)은 반드시 일치하지 않는다.**

예를 들어, South Africa에서 deep_persona의 WD가 가장 높지만(분포적으로 가장 다름), SFS에서는 가장 높은 structural fidelity를 보인다. 즉, "평균적으로는 틀리지만 항목 간 상관 구조는 잘 재현한다". 이는 persona의 상세도가 올라갈수록 내부 구조의 현실성이 높아지지만, 반드시 분포적 정확성까지 보장하지는 않음을 시사한다.

---

## 3. 프로젝트 구조

### 디렉토리 트리

```
synthetic_persona/
├── CLAUDE.md                         # Claude Code 가이드
├── README.md                         # 프로젝트 설명 (한국어)
├── pyproject.toml                    # Python 패키지 설정 (uv/pip)
├── uv.lock                           # 의존성 잠금 파일
├── main.py                           # CLI 진입점: survey/metrics/analyze/status
├── run_all.sh                        # 전체 파이프라인 원클릭 실행
├── .gitignore                        # 대용량 데이터/venv 제외
│
├── config/
│   └── experiment_config.py          # 모든 실험 설정의 단일 진실 원천 (Single Source of Truth)
│
├── prompts/                          # 3가지 Persona 프롬프팅 전략
│   ├── __init__.py                   # format_question_prompt(), run_survey() 공유 유틸
│   ├── cultural_prompting.py         # 국적 + 나이/성별만 (가장 단순)
│   ├── opencharacter_persona.py      # 인구통계 확률 샘플링 bio (중간)
│   └── deep_persona.py              # 7 anchor + taxonomy 기반 심층 persona (가장 상세)
│
├── engine/                           # LLM 추론 엔진
│   ├── llm_client.py                 # LocalLLM: vLLM 오프라인 추론 (배치 지원)
│   ├── run_survey.py                 # 실험 루프: persona 생성 → 배치 추론 → CSV 저장
│   └── vllm_server.py               # [Deprecated] 서버 방식 폐기
│
├── metrics/                          # 메트릭 계산
│   ├── step_a_gt_based.py            # WD, JSD, KS, MeanDiff (GT 필요)
│   ├── step_b_structural.py          # SFS = mean(SignF, SigF, NullF) (GT 필요)
│   ├── step_b_compute_all.py         # Step B 배치 러너
│   ├── step_c_gt_free.py             # SCS, VCR, ICE 계산 함수 (GT 불필요) ← 핵심
│   ├── step_c_compute_all.py         # Step C 배치 러너
│   └── analysis.py                   # Spearman/Kendall 순위 일치도 분석
│
├── scripts/                          # 파이프라인 bash 스크립트
│   ├── 01_start_vllm.sh              # 환경 확인 (vLLM import + 모델 캐시)
│   ├── 02_run_domain_b.sh            # WVS 실험 실행
│   ├── 03_run_domain_a.sh            # Privacy 실험 실행
│   ├── 04_compute_metrics.sh         # Step A + C 메트릭 계산
│   └── 05_analyze.sh                 # 분석 실행
│
├── data/
│   ├── wvs/                          # WVS Wave 7 원본 (190MB, git 제외)
│   ├── wvs_gt/                       # WVS GT: distributions.json + 국가별 CSV
│   ├── privacy_gt/                   # Privacy GT: distributions.json + 국가별 CSV
│   ├── preprocess_wvs.py             # WVS 원본 → distributions.json 변환
│   └── preprocess_privacy.py         # Privacy 원본 → distributions.json 변환
│
├── results/
│   ├── wvs/{country}/{method}.csv    # WVS 합성 응답 (6국 × 3방법 = 18 파일)
│   ├── privacy/{country}/{method}.csv # Privacy 합성 응답 (2국 × 3방법 = 6 파일)
│   └── metrics/                      # 메트릭 계산 결과 (JSON)
│       ├── step_a_results.json       # GT-based: WD/JSD/KS/MeanDiff (24 conditions)
│       ├── step_b_results.json       # Structural: SignF/SigF/NullF/SFS (24 conditions)
│       ├── step_c_results.json       # GT-free: SCS/VCR/ICE/DI_combined (24 conditions)
│       └── analysis_results.json     # 순위 일치도 분석 결과
│
└── reports_for_human/                # 사람이 읽는 리포트
    ├── progress_report.md            # 진행 상황 리포트
    └── POST_EXPERIMENT_PRD.md        # 이 파일
```

### 핵심 파일 5개 상세

#### (1) `config/experiment_config.py` — 실험의 단일 진실 원천

모든 실험 설정이 이 파일에 집중되어 있다. 다른 어떤 파일에도 하드코딩하지 않는다.

**주요 설정값**:
```python
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TENSOR_PARALLEL_SIZE = 2           # GPU 0,1 사용
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
TEMPERATURE = 0.7
NUM_RESPONSES_PER_CONDITION = 300
```

**주요 함수**: `get_all_conditions()` → 24개 `ExperimentCondition` 객체 리스트 생성
```python
@dataclass
class ExperimentCondition:
    domain: str           # "wvs" or "privacy"
    country: str          # "Argentina", "South Africa" 등
    prompt_method: str    # "cultural", "opencharacter", "deep_persona"
    items: dict           # WVS_ITEMS 또는 PRIVACY_ITEMS
    n_responses: int      # 300
```

#### (2) `engine/llm_client.py` — vLLM 오프라인 추론

**핵심 클래스**: `LocalLLM` — vLLM의 `LLM` 클래스를 래핑. 프로세스 내에서 모델을 로드하고, 프로세스 종료 시 GPU 자동 해제.

**핵심 메서드**:
```python
class LocalLLM:
    def query(self, system_prompt, user_prompt) -> str:
        """단일 chat completion"""

    def query_int(self, system_prompt, user_prompt, scale_min, scale_max) -> int | None:
        """단일 쿼리 후 정수 파싱"""

    def query_batch(self, conversations: list[list[dict]]) -> list[str]:
        """배치 chat completion — 핵심 성능 포인트"""

    def query_int_batch(self, conversations, scale_min, scale_max) -> list[int | None]:
        """배치 쿼리 후 정수 파싱"""
```

**입력 형식** (conversations):
```python
[
    [{"role": "system", "content": "You are a 58-year-old male citizen of Argentina..."},
     {"role": "user", "content": "For each of the following...\nScale: 1=Not very important..."}],
    [{"role": "system", "content": "You are a 19-year-old female citizen of Argentina..."},
     {"role": "user", "content": "For each of the following...\nScale: 1=Not very important..."}],
    # ... 300개
]
```

#### (3) `engine/run_survey.py` — 실험 루프

**핵심 함수**: `run_condition(llm, cond)` — 하나의 실험 조건을 실행

```
입력: LocalLLM 인스턴스, ExperimentCondition
과정:
  1) generate_persona(country, 300) → 300개 시스템 프롬프트
  2) for each item (6개):
       300개 conversation 생성 → llm.query_int_batch() → 300개 정수 응답
  3) DataFrame 구성 → CSV 저장
출력: results/{domain}/{country}/{method}.csv
```

**resume 기능**: 이미 존재하는 CSV는 건너뛴다. 중간에 중단되어도 이전 결과 유지.

#### (4) `metrics/step_c_gt_free.py` — GT-free 결함 지표 (핵심)

**3개 함수**: `scs(df)`, `vcr(df)`, `ice(df)` — 각각 SCS, VCR, ICE를 계산

**입력**: 합성 응답 DataFrame (300행 × 6열 형태)
```
respondent_id  Q45  Q46  Q57  Q184  Q218  Q254
0              3    2    2    7     3     1
1              3    2    2    6     3     1
...
```

**출력**: 각 지표의 raw값과 DI(Defect Index) 값
```python
{"alpha": -0.207, "DI_SCS": 0.907}
{"VCR": 0.35, "DI_VCR": 0.0}
{"ICE": 2.72, "ICE_normalized": 0.630, "DI_ICE": -0.630}
```

#### (5) `metrics/analysis.py` — 순위 일치도 분석

**핵심 함수**: `pooled_analysis(df)` — 전체 24개 조건의 Spearman ρ / Kendall τ 계산

```
입력: Step A (GT-based) + Step C (GT-free) 병합 DataFrame
출력: {
  "DI_ICE_vs_JSD_mean": {"spearman_r": 0.797, "spearman_p": 0.0000, ...},
  "DI_combined_vs_JSD_mean": {"spearman_r": 0.655, "spearman_p": 0.0005, ...},
  ...
}
```

### 데이터 흐름도

```
config/experiment_config.py
  │  (24 conditions: 8 countries × 3 methods)
  ▼
prompts/*.py  ──generate_persona()──→  300 system prompts per condition
  │
  ▼
engine/run_survey.py
  │  배치 추론: 6 items × 300 prompts = 1,800 queries per condition
  │  LocalLLM.query_int_batch() ──→ vLLM offline (GPU 0,1)
  ▼
results/{domain}/{country}/{method}.csv
  │  (300행 × 6열 정수 Likert 응답)
  ▼
┌─────────────────┬──────────────────────┬──────────────────────┐
│ metrics/         │ metrics/              │ metrics/              │
│ step_a_gt_based │ step_b_structural    │ step_c_gt_free       │
│ (GT 필요)        │ (GT 필요)             │ (GT 불필요) ← 핵심    │
│ WD, JSD, KS,    │ SignF, SigF, NullF,  │ SCS, VCR, ICE,      │
│ MeanDiff        │ SFS                  │ DI_combined          │
└────────┬────────┴──────────┬───────────┴──────────┬───────────┘
         │                   │                      │
         ▼                   ▼                      ▼
         metrics/analysis.py ──→ Spearman ρ, Kendall τ
         (GT-based 순위 vs GT-free 순위 일치?)
```

---

## 4. 환경 설정 및 재현 방법

### 필요한 하드웨어

| 항목 | 최소 | 권장 (이 실험에서 사용) |
|------|------|------------------------|
| GPU | NVIDIA GPU × 2, 각 16GB+ VRAM | RTX 3090 × 2 (24GB × 2) |
| RAM | 32GB | 64GB |
| 디스크 | 20GB 여유 | 50GB 여유 (모델 캐시 포함) |
| CUDA | 12.0+ | 13.0 (Driver 580.119.02) |
| Python | 3.11+ | 3.11 |

### 설치

```bash
cd /data/workspace/choie1/synthetic_persona

# (Option 1) uv 사용 — 권장
uv sync

# (Option 2) pip 사용
pip install -e .
```

주요 의존성: `vllm>=0.1.2`, `torch>=2.11.0`, `pandas>=3.0.1`, `scipy>=1.17.1`, `numpy>=2.4.3`

### 모델 다운로드

vLLM이 첫 실행 시 자동으로 HuggingFace에서 다운로드한다. 미리 캐시하려면:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct')
print('Model cached.')
"
```

### WVS Wave 7 데이터 준비

1. WVS 공식 사이트 (https://www.worldvaluessurvey.org/) 에서 Wave 7 CSV 다운로드
2. `data/wvs/` 디렉토리에 저장
3. 전처리 실행:

```bash
python data/preprocess_wvs.py
# 출력: data/wvs_gt/distributions.json + 국가별 CSV
```

이미 전처리된 파일이 `data/wvs_gt/`에 포함되어 있으므로, 재현 시 이 단계는 건너뛸 수 있다.

### 전체 실험 재현 — 원라이너

```bash
cd /data/workspace/choie1/synthetic_persona && bash run_all.sh
```

이 명령은 순서대로: 환경 확인 → WVS 실험(18 conditions) → Privacy 실험(6 conditions) → 메트릭 계산 → 분석 → git commit을 실행한다.

### 개별 실행 (단계별)

```bash
# 환경 확인
bash scripts/01_start_vllm.sh
# 기대 출력: "[01] vLLM import OK" + "[01] Environment ready."

# WVS 실험 (약 5-7분)
CUDA_VISIBLE_DEVICES=0,1 python -m engine.run_survey --domain wvs
# 기대 출력: 18개 조건의 tqdm 진행률 바 + "Domain 'wvs' complete."
# 결과 파일: results/wvs/{country}/{method}.csv (18개 파일, 각 301줄)

# Privacy 실험 (약 2분)
CUDA_VISIBLE_DEVICES=0,1 python -m engine.run_survey --domain privacy
# 기대 출력: 6개 조건 + "Domain 'privacy' complete."
# 결과 파일: results/privacy/{country}/{method}.csv (6개 파일)

# 메트릭 계산
python -m metrics.step_a_gt_based    # Step A: GT-based
python -m metrics.step_c_compute_all  # Step C: GT-free
python -m metrics.step_b_compute_all  # Step B: Structural

# 분석
python -m metrics.analysis --metrics-dir results/metrics
# 기대 출력: 순위 일치도 테이블 + Spearman ρ 테이블

# 파이프라인 상태 확인
python main.py status
```

**주의**: 이 프로젝트의 venv를 명시적으로 사용해야 할 수 있다:
```bash
# 다른 venv가 활성화된 경우
/data/workspace/choie1/synthetic_persona/.venv/bin/python -m engine.run_survey --domain wvs
```

---

## 5. 실험 설계

### 실험 조건 매트릭스

| | Cultural | OpenCharacter | DeepPersona |
|---|---|---|---|
| **WVS: Argentina** | 300 responses | 300 responses | 300 responses |
| **WVS: Australia** | 300 | 300 | 300 |
| **WVS: Germany** | 300 | 300 | 300 |
| **WVS: India** | 300 | 300 | 300 |
| **WVS: Kenya** | 300 | 300 | 300 |
| **WVS: United States** | 300 | 300 | 300 |
| **Privacy: South Africa** | 300 | 300 | 300 |
| **Privacy: United Kingdom** | 300 | 300 | 300 |

**총**: 8 countries × 3 methods × 300 responses = **7,200 합성 응답**
**총 LLM 쿼리**: 7,200 × 6 items = **43,200 queries** (배치 추론으로 144 batches)

### 각 실험 조건에서 일어나는 일

하나의 조건 (예: WVS / Argentina / deep_persona)의 실행 과정:

1. `deep_persona.generate_persona("Argentina", 300, seed=42)` → 300개의 고유한 시스템 프롬프트 생성
2. 6개 질문(Q45, Q46, Q57, Q184, Q218, Q254) 각각에 대해:
   - 300개의 conversation 구성: `[{"role":"system", persona_i}, {"role":"user", question}]`
   - `LocalLLM.query_int_batch(conversations, scale_min, scale_max)` → 300개 정수 응답
3. 300 × 6 DataFrame → `results/wvs/Argentina/deep_persona.csv`

### WVS 6개 질문 원문과 스케일

| 코드 | 질문 원문 | 스케일 | 라벨 |
|------|-----------|--------|------|
| Q45 | For each of the following, indicate how important it is in your life: **Family** | 1-3 | 1=Not very important, 2=Rather important, 3=Very important |
| Q46 | For each of the following, indicate how important it is in your life: **Friends** | 1-4 | 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all important |
| Q57 | Generally speaking, would you say that most people can be trusted or that you need to be very careful in dealing with people? | 1-2 | 1=Most people can be trusted, 2=Need to be very careful |
| Q184 | All things considered, how satisfied are you with your life as a whole these days? | 1-10 | 1=Completely dissatisfied, 10=Completely satisfied |
| Q218 | Please tell me whether you think the following can always be justified, never be justified, or something in between: **Homosexuality** | 1-3 | 1=Never justifiable, 2=Something in between, 3=Always justifiable |
| Q254 | Here is a list of qualities that children can be encouraged to learn at home. How important is **'Tolerance and respect for other people'**? | 1-5 | 1=Very important ... 5=Not important at all |

**주의**: Q45(1-3)와 Q57(1-2)는 스케일 범위가 매우 좁아, temperature=0.7에서도 cultural prompting에서 분산이 0인 경우가 있다.

### Privacy Calculus 6개 질문

| 코드 | 질문 원문 | 스케일 |
|------|-----------|--------|
| PC1 | Sharing my personal data with online services provides me with significant benefits | 1-7 (Strongly disagree~Strongly agree) |
| PC2 | I am concerned that my personal data could be misused by online services | 1-7 |
| PC3 | I trust that online service providers will protect my personal information | 1-7 |
| PC4 | I consider my personal information to be highly sensitive | 1-7 |
| PC5 | I am willing to share my personal data with online services in exchange for benefits | 1-7 |
| PC6 | I am concerned about my privacy when using online services | 1-7 |

### 하이퍼파라미터

| 파라미터 | 값 | 설정 위치 |
|----------|-----|-----------|
| Model | Qwen/Qwen2.5-3B-Instruct | `config/experiment_config.py` |
| Temperature | 0.7 | `config/experiment_config.py` |
| Max tokens | 512 | `config/experiment_config.py` |
| Tensor parallel size | 2 | `config/experiment_config.py` |
| GPU memory utilization | 0.90 | `config/experiment_config.py` |
| Max model length | 2048 | `config/experiment_config.py` |
| dtype | auto (bfloat16) | `engine/llm_client.py` |
| enforce_eager | True | `engine/llm_client.py` |
| Persona seed | 42 | `engine/run_survey.py` → `generate_persona(seed=42)` |
| Responses per condition | 300 | `config/experiment_config.py` |

---

## 6. 실험 결과

### 6.1. GT-based 메트릭 전체 테이블 (Step A)

`results/metrics/step_a_results.json` 기반:

| Country | Method | WD | JSD | KS | MeanDiff |
|---------|--------|-----|------|------|----------|
| Argentina | cultural | 1.6399 | 0.4333 | 0.7541 | 1.5611 |
| Argentina | opencharacter | 1.0382 | 0.2723 | 0.5316 | 0.9288 |
| Argentina | deep_persona | **0.6451** | **0.1917** | **0.4498** | **0.5780** |
| Australia | cultural | 1.3300 | 0.3751 | 0.6216 | 0.9968 |
| Australia | opencharacter | 0.8867 | 0.2486 | 0.4171 | 0.4880 |
| Australia | deep_persona | **0.8048** | **0.1614** | **0.4069** | 0.7834 |
| Germany | cultural | 1.3720 | 0.3809 | 0.6664 | 1.1445 |
| Germany | opencharacter | 0.8701 | 0.2434 | 0.4328 | 0.5889 |
| Germany | deep_persona | **0.6377** | **0.1723** | **0.3829** | **0.5363** |
| India | cultural | 1.7090 | 0.4198 | 0.7428 | 1.6734 |
| India | opencharacter | 1.2065 | 0.3400 | 0.6370 | 1.0956 |
| India | deep_persona | **0.7543** | **0.1800** | **0.4461** | **0.7179** |
| Kenya | cultural | 1.4607 | 0.3922 | 0.6666 | 1.3162 |
| Kenya | opencharacter | 1.0559 | 0.2956 | 0.5475 | 0.8684 |
| Kenya | deep_persona | **0.6852** | **0.1859** | **0.4471** | **0.6057** |
| United States | cultural | 1.2560 | 0.3379 | 0.6298 | 1.0864 |
| United States | opencharacter | 0.8500 | 0.2291 | 0.4171 | 0.5591 |
| United States | deep_persona | **0.5987** | **0.1492** | **0.3983** | 0.5641 |
| South Africa | cultural | 1.5572 | 0.2798 | 0.4789 | 1.1917 |
| South Africa | opencharacter | **1.2772** | 0.2581 | 0.5067 | **0.8628** |
| South Africa | deep_persona | 2.0833 | **0.2527** | **0.4483** | 1.8600 |
| United Kingdom | cultural | 2.0958 | 0.3868 | 0.6374 | 1.7911 |
| United Kingdom | opencharacter | 1.7078 | 0.3087 | 0.5438 | 1.3690 |
| United Kingdom | deep_persona | **1.5053** | **0.1966** | **0.4216** | 1.4400 |

**일관된 패턴 (WVS 6개국)**: cultural > opencharacter > deep_persona (WD 기준)
→ **DeepPersona가 모든 WVS 국가에서 GT에 가장 근접**

**예외 (Privacy)**: South Africa에서 deep_persona의 WD가 역전 (2.08 — 가장 높음). UK GT 데이터(~17명)의 신뢰도 문제 가능성.

### 6.2. GT-free 결함 지표 전체 테이블 (Step C)

`results/metrics/step_c_results.json` 기반:

| Country | Method | α | DI_SCS | VCR | DI_VCR | ICE_norm | DI_ICE | **DI_comb** |
|---------|--------|------|--------|------|--------|----------|--------|-------------|
| Argentina | cultural | 0.024 | 0.676 | — | 0.0 | 0.444 | -0.444 | 0.324 |
| Argentina | opencharacter | -0.354 | 1.054 | — | 0.0 | 0.415 | -0.415 | 0.452 |
| Argentina | deep_persona | -0.207 | 0.907 | — | 0.0 | 0.630 | -0.630 | 0.245 |
| Australia | cultural | 0.079 | 0.621 | — | 0.0 | 0.213 | -0.213 | 0.486 |
| Australia | opencharacter | -0.475 | 1.175 | — | 0.0 | 0.549 | -0.549 | 0.382 |
| Australia | deep_persona | -0.349 | 1.049 | — | 0.0 | 0.645 | -0.645 | 0.273 |
| Germany | cultural | -0.086 | 0.786 | — | 0.0 | 0.213 | -0.213 | 0.533 |
| Germany | opencharacter | -0.567 | 1.267 | — | 0.0 | 0.439 | -0.439 | 0.493 |
| Germany | deep_persona | -0.480 | 1.180 | — | 0.0 | 0.566 | -0.566 | 0.350 |
| India | cultural | -0.016 | 0.716 | — | 0.0 | 0.213 | -0.213 | 0.513 |
| India | opencharacter | -0.248 | 0.948 | — | 0.0 | 0.444 | -0.444 | 0.400 |
| India | deep_persona | -0.453 | 1.153 | — | 0.0 | 0.603 | -0.603 | 0.334 |
| Kenya | cultural | -0.006 | 0.706 | — | 0.0 | 0.213 | -0.213 | 0.510 |
| Kenya | opencharacter | -0.126 | 0.826 | — | 0.0 | 0.338 | -0.338 | 0.447 |
| Kenya | deep_persona | -0.446 | 1.146 | — | 0.0 | 0.612 | -0.612 | 0.325 |
| United States | cultural | -0.010 | 0.710 | — | 0.0 | 0.213 | -0.213 | 0.511 |
| United States | opencharacter | -0.151 | 0.851 | — | 0.0 | 0.500 | -0.500 | 0.329 |
| United States | deep_persona | -0.262 | 0.962 | — | 0.0 | 0.571 | -0.571 | 0.306 |
| South Africa | cultural | -0.021 | 0.721 | — | 0.0 | 0.225 | -0.225 | 0.505 |
| South Africa | opencharacter | 0.136 | 0.564 | — | 0.0 | 0.295 | -0.295 | 0.407 |
| South Africa | deep_persona | **0.627** | **0.073** | — | 0.0 | 0.603 | -0.603 | **0.033** |
| United Kingdom | cultural | -0.058 | 0.758 | — | 0.0 | 0.286 | -0.286 | 0.468 |
| United Kingdom | opencharacter | 0.065 | 0.635 | — | 0.0 | 0.285 | -0.285 | 0.435 |
| United Kingdom | deep_persona | **0.617** | **0.083** | — | 0.0 | 0.611 | -0.611 | **0.030** |

**DI_VCR = 0.0 전체**: 모든 조건에서 VCR < 0.5 → 현재 실험에서 변별력 없음.

**주목할 결과**: Privacy 도메인의 deep_persona에서 α가 0.62-0.63으로 건강 범위(0.5-0.9)에 진입 → DI_SCS가 0.07-0.08로 매우 낮음 → DI_combined이 0.03. 이는 Privacy 항목이 모두 7점 Likert로 uniform하여, 충분히 상세한 persona(DeepPersona)가 현실적인 내적 일관성을 재현할 수 있음을 시사.

### 6.3. 핵심 검증: DI 순위 vs GT-based 순위

#### Pooled Spearman ρ (24 conditions)

`results/metrics/analysis_results.json` 기반:

|  | **DI_combined** | **DI_SCS** | **DI_VCR** | **DI_ICE** |
|---|---|---|---|---|
| **WD_mean** | +0.355 (p=.089) | -0.768*** | N/A | **+0.558*** |
| **JSD_mean** | **+0.655***  | -0.528** | N/A | **+0.797***  |
| **KS_mean** | **+0.596**  | -0.502* | N/A | **+0.716***  |
| **MeanDiff** | +0.243 (n.s.) | -0.743*** | N/A | +0.427* |

(\* p<0.05, \*\* p<0.01, \*\*\* p<0.001)

**해석**:
- **DI_ICE ↔ JSD: ρ = +0.797 (p < 0.0001)** — 매우 강한 양의 상관. ICE가 GT-free 지표로서 가장 유효.
- **DI_combined ↔ JSD: ρ = +0.655 (p = 0.0005)** — 유의미한 양의 상관. 복합 지표도 JSD 기준 유효.
- **DI_SCS ↔ WD: ρ = -0.768** — 강한 음의 상관. α가 낮을수록(DI_SCS가 낮을수록) WD가 높아지는 역방향 관계. DI_SCS는 DI_combined에서 역효과를 냄.

#### Per-country 순위 일치 (DI_combined vs WD rank)

| 국가 | WD rank | DI_combined rank | 일치? |
|------|---------|-----------------|-------|
| Argentina | cultural>OC>DP | OC>cultural>DP | **NO** |
| Australia | cultural>OC>DP | cultural>OC>DP | **YES** |
| Germany | cultural>OC>DP | cultural>OC>DP | **YES** |
| India | cultural>OC>DP | cultural>OC>DP | **YES** |
| Kenya | cultural>OC>DP | cultural>OC>DP | **YES** |
| United States | cultural>OC>DP | cultural>OC>DP | **YES** |
| South Africa | DP>cultural>OC | cultural>OC>DP | **NO** |
| United Kingdom | cultural>OC>DP | cultural>OC>DP | **YES** |

**일치율: 6/8 (75%)** — 우연 기대값(33%)을 크게 상회.

### 6.4. Structural Fidelity (Step B) 결과

`results/metrics/step_b_results.json` 기반 (WVS 국가 평균):

| Method | SignF (평균) | SigF (평균) | NullF (평균) | SFS (평균) |
|--------|------------|------------|-------------|-----------|
| Cultural | 0.111 | 0.019 | 0.096 | 0.075 |
| OpenCharacter | 0.345 | 0.190 | 0.225 | 0.253 |
| DeepPersona | **0.434** | **0.608** | **0.459** | **0.500** |

**일관된 패턴**: deep_persona > opencharacter > cultural — persona가 상세할수록 GT의 상관 구조를 더 잘 재현.

### 6.5. 성공/실패 판단

| 가설 | 기준 | 결과 | 판정 |
|------|------|------|------|
| H1a: DI_ICE ↔ JSD ρ > 0.5 | ρ=0.797, p<0.001 | **지지** |
| H1b: DI_combined ↔ JSD ρ > 0.5 | ρ=0.655, p=0.0005 | **지지** |
| H1c: Per-country 일치율 > 33% | 75% (6/8) | **지지** |

---

## 7. 알려진 문제점 및 트러블슈팅

### 7.1. Cultural Prompting NaN 문제

**증상**: Cultural prompting의 300개 응답이 모두 동일한 값 → 분산=0 → Cronbach's α = NaN → SCS, VCR, ICE 전부 NaN → DI_combined = 0.0 (실제로는 최악인데 0으로 표시)

**원인**: `temperature=0.0` + 300개 동일 프롬프트(이전 버전에서는 국적만 부여, age/gender 없음) → LLM이 결정론적으로 동일한 응답 생성.

**해결 방법** (2단계):
1. `prompts/cultural_prompting.py`에 나이+성별 variation 추가 → 각 persona가 고유한 프롬프트
2. `config/experiment_config.py`에서 `TEMPERATURE = 0.0` → `0.7`로 변경

**교훈**: temperature=0.0은 합성 설문 연구에 부적합. 짧은 스케일(1-2, 1-3)에서는 temp=0.7에서도 분산이 0일 수 있다.

**확인 방법**:
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/wvs/Argentina/cultural.csv')
for c in [col for col in df.columns if col.startswith('Q')]:
    print(f'{c}: unique={df[c].nunique()}, std={df[c].std():.4f}')
"
```
정상이면 최소 2개 이상의 unique 값, std > 0 (Q184 등 넓은 스케일에서 확인).

### 7.2. vLLM 서버 → 오프라인 모드 전환

**증상**: vLLM 서버를 백그라운드로 띄우면, 세션이 끊겨도 GPU를 계속 점유. `nvidia-smi`에서 VLLM::Worker_TP0/TP1이 22GB씩 GPU 0,1을 점유. `kill -9`로만 종료 가능.

**원인**: vLLM 서버는 독립 프로세스. 실행 스크립트와 생명주기가 분리됨.

**해결**: vLLM 서버 방식을 완전 폐기하고, `vllm.LLM` 클래스를 프로세스 내에서 직접 사용하는 오프라인 모드로 전환.

```python
# Before (서버 방식):
# 1) 서버 시작: python -m vllm.entrypoints.openai.api_server ...
# 2) HTTP 요청: requests.post("http://localhost:8000/v1/chat/completions", ...)
# 3) 서버 종료: kill $PID

# After (오프라인 방식):
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", tensor_parallel_size=2, ...)
outputs = llm.chat(conversations, SamplingParams(temperature=0.7, max_tokens=512))
# 프로세스 종료 시 GPU 자동 해제
```

**추가 이점**: 배치 추론이 네이티브하게 지원되어, 300개 프롬프트를 한 번에 처리 가능 (이전: 순차 HTTP 요청).

**GPU 상태 확인**:
```bash
nvidia-smi
# 기대: "No running processes found" (실험 실행 중이 아닐 때)
```

**만약 GPU 좀비가 발견되면**:
```bash
kill -9 $(pgrep -f vllm) 2>/dev/null
# 또는 PID 직접 지정
nvidia-smi  # 확인
```

### 7.3. Python venv 충돌

**증상**: `ModuleNotFoundError: No module named 'scipy'` 또는 `AttributeError: _ARRAY_API not found` (pyarrow/numpy 충돌)

**원인**: 다른 프로젝트의 venv(`csmed_agent`)가 PATH에 우선적으로 잡힘.

**해결**: 이 프로젝트의 venv를 명시적으로 사용:
```bash
/data/workspace/choie1/synthetic_persona/.venv/bin/python -m metrics.step_a_gt_based
```

### 7.4. DI_SCS 역방향 상관

**증상**: DI_SCS가 GT-based 메트릭과 음의 상관 (ρ = -0.768). cultural prompting은 DI_SCS가 가장 낮지만(α가 0에 가까워 |α-0.7|이 작음), GT에서는 가장 나쁨.

**원인**: cultural은 variation이 매우 낮아 α가 0 근처 → |α - 0.7| ≈ 0.7. 한편 deep_persona는 음의 α(역상관 패턴) → |α - 0.7| > 1.0. 즉, DI_SCS 공식이 "variation 부족"과 "역상관 패턴"을 구별하지 못함.

**잠재적 해결**: DI_SCS 공식을 비대칭으로 변경하거나, DI_combined에서 SCS를 제외하고 ICE만 사용.

---

## 8. 남은 작업

### 우선순위 1: 논문 작성에 즉시 필요

1. **DI_SCS 공식 개선**: 현재 |α - 0.7|은 역방향 상관을 야기. α < 0 케이스에 별도 패널티 부여, 또는 단방향 공식 (max(0, α - 0.9) + max(0, 0.3 - α)) 검토
2. **DI_VCR 유효성 검증**: 현재 모든 조건에서 DI_VCR=0. threshold(0.5)를 낮추거나, 더 큰 모델(7B, 13B)에서 halo effect가 나타나는지 추가 실험
3. **DI_combined 가중치 최적화**: 현재 SCS:VCR:ICE = 1:1:1. ICE만 사용하거나, SCS 가중치를 줄이는 방안

### 우선순위 2: 실험 확장

4. **더 큰 모델 비교**: Qwen 7B, 14B 또는 Llama 3 8B에서 동일 실험. 모델 크기가 DI에 미치는 영향 분석
5. **Temperature 민감도 분석**: 0.3, 0.5, 0.7, 1.0에서 각각 실험하여 temperature가 DI에 미치는 영향 정량화
6. **추가 국가/도메인**: Privacy Calculus에서 더 많은 국가 추가. 다른 설문 도메인(예: Big Five 성격) 테스트

### 우선순위 3: 방법론 강화

7. **Step B 배치 러너 정식화**: `metrics/step_b_compute_all.py`를 04_compute_metrics.sh에 통합
8. **통계적 검정력 분석**: 300 responses가 충분한지, 500이나 1000이 필요한지 분석
9. **Privacy GT 데이터 보강**: UK는 ~17명으로 GT 자체의 신뢰도가 낮음. 더 큰 샘플 수집 필요

---

## 9. 논문 연결

### 예상 테이블/Figure 매핑

| 논문 요소 | 데이터 출처 | 비고 |
|-----------|------------|------|
| Table 1: 실험 설계 | `config/experiment_config.py` | 8국 × 3방법 × 300 응답 |
| Table 2: GT-based 메트릭 | `results/metrics/step_a_results.json` | 본 PRD §6.1 |
| Table 3: GT-free DI 지표 | `results/metrics/step_c_results.json` | 본 PRD §6.2 |
| Table 4: Spearman ρ 매트릭스 | `results/metrics/analysis_results.json` | 본 PRD §6.3 |
| Table 5: SFS 결과 | `results/metrics/step_b_results.json` | 본 PRD §6.4 |
| Figure 1: DI_ICE vs JSD 산점도 | step_a + step_c 병합 | ρ=0.797 시각화 |
| Figure 2: Per-country 순위 비교 | analysis_results.json per_country | 히트맵 형태 |

### 가설별 지지/기각

| 가설 | 결과 | 논문에서의 주장 |
|------|------|----------------|
| **H1a**: DI_ICE ↔ JSD 양의 상관 | ρ=0.797*** | **강하게 지지**. ICE가 GT-free 품질 지표로 유효함을 주장 가능 |
| **H1b**: DI_combined ↔ JSD 양의 상관 | ρ=0.655*** | **지지**. 복합 지표도 유효하나, ICE 단독보다 약함 |
| **H1c**: Per-country 일치율 > 우연 | 75% (6/8) | **지지**. 단, 소수 불일치 사례(Argentina, South Africa) 분석 필요 |

### 논문에서 주장할 수 있는 것

1. **ICE는 GT 없이 합성 설문 데이터의 품질을 유의미하게 예측한다** (JSD 기준 ρ=0.797***)
2. **Persona의 상세도가 높을수록 GT에 근접한다** (WVS 6개국에서 일관됨)
3. **구조적 결함 진단은 분포적 유사성과 독립적 가치가 있다** (SFS 결과)

### 논문에서 주장할 수 없는 것

1. ~~DI_VCR이 halo effect를 탐지한다~~ (현재 데이터에서 변별력 없음)
2. ~~DI_SCS가 독립적으로 품질을 예측한다~~ (역방향 상관 문제)
3. ~~결과가 모든 LLM에 일반화된다~~ (Qwen 3B 단일 모델 실험)
4. ~~Privacy 도메인에서도 동일한 패턴~~ (GT 부족, deep_persona 역전)

---

## 10. 참고 문헌

### 핵심 참고 논문

1. **DeepPersona**: Wang, Y., et al. (2025). "DeepPersona: Progressively Personalized Synthetic Survey Generation Using LLMs." NeurIPS 2025. — 7 anchor attribute + taxonomy 기반 progressive persona generation. 본 연구의 deep_persona 프롬프팅 전략의 기반.

2. **Anthology**: Moon, S., et al. (2024). "Anthology: Generating Diverse Personas from Survey Data." EMNLP 2024. — Census 기반 persona bio 생성. 본 연구의 OpenCharacter 프롬프팅 전략에 영향.

3. **Cultural Prompting**: Tao, Y., et al. (2024). "Cultural Alignment in Large Language Models: An Explanatory Analysis Based on Hofstede's Cultural Dimensions." arXiv:2309.12342. — 국적만 부여하는 가장 단순한 프롬프팅 전략.

4. **Lutz et al.**: Lutz, B., et al. (2025). "Synthetic Survey Data: A Review of Challenges, Approaches, and Opportunities." EMNLP 2025 Findings. — 합성 설문 데이터의 품질 평가 프레임워크. distribution-level vs structure-level 평가의 구분.

### 메트릭 관련

5. **Cronbach's α**: Cronbach, L.J. (1951). "Coefficient alpha and the internal structure of tests." Psychometrika, 16(3), 297-334.

6. **Jensen-Shannon Divergence**: Lin, J. (1991). "Divergence measures based on the Shannon entropy." IEEE Transactions on Information Theory, 37(1), 145-151.

7. **Wasserstein Distance**: Villani, C. (2009). Optimal Transport: Old and New. Springer.

### 데이터

8. **WVS Wave 7**: Haerpfer, C., et al. (2022). World Values Survey Wave 7 (2017-2022). JD Systems Institute & WVSA Secretariat.
