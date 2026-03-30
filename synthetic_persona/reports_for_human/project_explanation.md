# Synthetic Persona Survey 프로젝트 완전 해설서

> 생성일: 2026-03-29 | Claude Code 대화 기반 정리

---

## 1. 프로젝트 목적

### 해결하려는 문제

LLM(대규모 언어 모델)에게 "너는 한국의 30대 직장인이야"라고 역할을 부여하고 설문에 답하게 하면, 실제 인간의 응답 데이터를 **합성(synthetic)**할 수 있습니다. 이런 방법은 설문 비용을 줄이고, 접근이 어려운 인구집단의 데이터를 생성할 수 있어 주목받고 있습니다.

**문제는: 그 합성 데이터의 품질을 어떻게 검증하느냐?**

지금까지의 방법은 "진짜 인간 데이터(Ground Truth, GT)"와 비교해서 얼마나 비슷한지를 확인했습니다. 하지만 그러려면 **이미 진짜 데이터가 있어야** 합니다. 진짜 데이터가 있으면 굳이 합성 데이터를 만들 필요가 없죠 — 이것이 **닭이 먼저냐 달걀이 먼저냐** 문제입니다.

### 비유

> 학생이 시험 답안을 제출했는데, **정답지 없이도** 그 답안이 엉터리인지 판별할 수 있을까?
>
> - 모든 문항에 같은 번호만 찍었다면? → 의심스럽다 (SCS가 잡아냄)
> - 전혀 다른 유형의 문항인데 답이 전부 비슷하다면? → 제대로 읽지 않은 것 같다 (VCR이 잡아냄)
> - 문항 간 답변 패턴에 다양성이 없다면? → 복사/붙여넣기 같다 (ICE가 잡아냄)
>
> 이 프로젝트는 **"정답지 없이 채점하는 방법"**을 만드는 것입니다.

### 논문에서 주장하고 싶은 것

**"SCS, VCR, ICE라는 3가지 GT-free 지표만으로도, 기존의 GT-based 지표(WD, JSD 등)와 동일한 품질 순위를 매길 수 있다."**

즉: GT-free 지표가 "이 시스템이 더 좋다"고 판단한 순위가, 실제 인간 데이터와 비교한 순위와 일치한다면 → GT 없이도 합성 데이터 품질 평가가 가능하다는 것을 증명한 셈입니다.

---

## 2. 전체 실험 흐름

### 큰 그림 흐름도

```
[1단계] vLLM 서버 시작 (Qwen 2.5-3B 모델 로드)
   ↓
[2단계] 설문 시뮬레이션 (6개국 × 3개 방식 × 300명 = 5,400명의 가상 응답 생성)
   ↓
   │  각 "가상 응답자"마다:
   │    (a) persona 시스템 프롬프트 생성 ("너는 아르헨티나의 35세 여성 교사야...")
   │    (b) 6개 설문 질문을 하나씩 던짐 ("가족이 얼마나 중요한가? 1-3 중 답해")
   │    (c) LLM이 정수 하나로 응답 → CSV에 기록
   ↓
[3단계] GT-free 메트릭 계산 (인간 데이터 필요 없음)
   │  각 조건(국가×방식)의 300명 응답 CSV에 대해:
   │    - SCS: 내적 일관성(α)이 건강 범위(0.7)에서 얼마나 벗어났나?
   │    - VCR: 모든 답변이 하나의 요인으로 수렴하나? (halo effect)
   │    - ICE: 문항 간 상관관계가 다양한가?
   │    → DI_combined = 세 지표의 정규화 평균 (낮을수록 좋음)
   ↓
[4단계] GT-based 메트릭 계산 (WVS 실제 인간 데이터와 비교)
   │  distributions.json의 실제 응답 분포 vs 합성 응답:
   │    - WD (Wasserstein Distance): 분포 간 "이동 비용"
   │    - JSD (Jensen-Shannon Divergence): 분포 유사도
   │    - KS (Kolmogorov-Smirnov): 누적분포 최대 차이
   │    - Mean Diff: 평균 차이
   ↓
[5단계] 순위 일치도 분석 ← ★ 핵심 검증
   │  "GT-free 순위 == GT-based 순위?" 확인
   │    - 국가별: 3개 방식의 WD 순위 vs DI 순위가 같은가?
   │    - 전체: 18개 조건의 Spearman 상관계수 (ρ)
   ↓
[결과] 논문 Table/Figure 생성
```

### `run_all.sh` 실행 시 내부 흐름

```bash
bash scripts/01_start_vllm.sh    # GPU에 Qwen 모델 올림, health check 대기 (최대 5분)
bash scripts/02_run_domain_b.sh  # python -m engine.run_survey --domain wvs
                                 #   → results/wvs/{국가}/{방식}.csv 18개 생성
bash scripts/03_run_domain_a.sh  # python -m engine.run_survey --domain privacy
                                 #   → results/privacy/{국가}/{방식}.csv 6개 생성
bash scripts/04_compute_metrics.sh  # Step A (GT-based) + Step C (GT-free)
                                    #   → results/metrics/step_a_results.json
                                    #   → results/metrics/step_c_results.json
bash scripts/05_analyze.sh       # Spearman ρ 분석
                                 #   → results/metrics/analysis_results.json
```

### 데이터 흐름 — 입력과 출력

| 단계 | 입력 | 처리 | 출력 |
|------|------|------|------|
| 01 | Qwen 모델 가중치 | vLLM이 GPU에 모델 로드 | localhost:8000 API 서버 |
| 02 | experiment_config.py (24개 조건) | 조건별 300명 × 6문항 LLM 호출 | `results/wvs/{국가}/{방식}.csv` |
| 04-StepA | distributions.json + 합성CSV | WD, JSD, KS, MeanDiff 계산 | `step_a_results.json` |
| 04-StepC | 합성CSV만 | SCS, VCR, ICE 계산 | `step_c_results.json` |
| 05 | step_a + step_c JSON | Spearman/Kendall 상관 | `analysis_results.json` |

---

## 3. 코드 구조 설명

### 설정

| 파일 | 역할 | 핵심 함수/변수 |
|------|------|----------------|
| `config/experiment_config.py` | 실험의 모든 설정을 한 곳에 모아둔 "설계도" — 국가 목록, 질문 목록(스케일 포함), 모델 ID, 경로 | `WVS_ITEMS`, `WVS_COUNTRIES`, `ExperimentCondition`, `get_all_conditions()` → 24개 조건 리스트 반환 |

### 엔진 (LLM 호출 + 설문 실행)

| 파일 | 역할 | 핵심 함수 |
|------|------|-----------|
| `engine/vllm_server.py` | vLLM 서버 프로세스 시작/중지/헬스체크 | `start_server()`, `stop_server()` |
| `engine/llm_client.py` | vLLM API를 호출하는 클라이언트 — "1~10 사이 정수 하나 답해"를 보내고 정수를 파싱 | `query_int(system_prompt, user_prompt, scale_min, scale_max)` → `int` |
| `engine/run_survey.py` | 설문 시뮬레이션 메인 루프 — 조건별로 persona 생성 → 질문 전송 → CSV 저장 | `run_condition(client, cond)` → `DataFrame`, `run_domain("wvs")` |

### 프롬프트 (3가지 persona 전략)

| 파일 | 역할 | 핵심 함수 |
|------|------|-----------|
| `prompts/__init__.py` | 질문 포맷팅 유틸리티 | `format_question_prompt(item_dict)` → `str` |
| `prompts/cultural_prompting.py` | 국적만 부여하는 가장 단순한 방식 | `generate_persona(country, n)` → 동일 프롬프트 n개 |
| `prompts/opencharacter_persona.py` | 나이/성별/직업/종교 등 인구통계 샘플링 → 3문장 bio | `generate_persona(country, n, seed)` → 각각 다른 프롬프트 n개 |
| `prompts/deep_persona.py` | 7개 앵커 + 21개 taxonomy → Big Five 성격, 가치관, 라이프스타일 풀 프로필 | `generate_persona(country, n, seed)` → 상세 프로필 n개 |

### 메트릭 (품질 측정)

| 파일 | 역할 | 핵심 함수 |
|------|------|-----------|
| `metrics/step_a_gt_based.py` | **GT 필요** — 합성 vs 실제 분포 비교 (WD, JSD, KS, MeanDiff) | `run_step_a()` → `step_a_results.json` 저장 |
| `metrics/step_b_structural.py` | **GT 필요** — 상관 구조 충실도 (SignF, SigF, NullF → SFS) | `compute_sfs(syn, real)` → `dict` |
| `metrics/step_c_gt_free.py` | **GT 불필요 ★** — SCS, VCR, ICE 산출 (핵심 기여) | `scs(df)`, `vcr(df)`, `ice(df)`, `compute_combined_di(all_metrics)` |
| `metrics/step_c_compute_all.py` | Step C를 전체 조건에 배치 실행 | `run_step_c()` → `step_c_results.json` 저장 |
| `metrics/analysis.py` | GT-free 순위 vs GT-based 순위의 Spearman ρ 계산 | `per_country_analysis(df)`, `pooled_analysis(df)`, `run_analysis()` |

### 스크립트 (파이프라인 단계별)

| 스크립트 | 역할 |
|----------|------|
| `scripts/01_start_vllm.sh` | vLLM 서버 시작, GPU 0,1 사용, health check 후 smoke test |
| `scripts/02_run_domain_b.sh` | `python -m engine.run_survey --domain wvs` 실행 |
| `scripts/03_run_domain_a.sh` | `python -m engine.run_survey --domain privacy` 실행 |
| `scripts/04_compute_metrics.sh` | Step A (GT-based) + Step C (GT-free) 메트릭 계산 |
| `scripts/05_analyze.sh` | 순위 일치도 분석 실행 |

---

## 4. 3가지 Persona 시스템 비교

### 한눈에 비교

| | Cultural | OpenCharacter | DeepPersona |
|---|---|---|---|
| **정보량** | 국적 1줄 | 인구통계 3문장 | 풀 프로필 ~20줄 |
| **다양성** | 300명 전부 동일 | 나이/성별/직업 등 샘플링 | Big Five + 가치관 + 라이프스타일 |
| **비유** | "한국인 대표" | "30대 여성 교사" | "30대, 외향적, 진보적, 채식주의, 가톨릭 여성 교사" |

### 실제 프롬프트 예시

**Cultural Prompting** — Argentina의 경우:
```
[System] You are a citizen of Argentina. Please answer the following survey
question from the perspective of a typical person living in Argentina.
Respond with ONLY a single integer number on the given scale.
Do not add any explanation or reasoning.

[User] For each of the following, indicate how important it is in your life: Family

Scale: 1=Not very important, 2=Rather important, 3=Very important

Answer with ONLY a single integer from 1 to 3.
```

**OpenCharacter** — 샘플링된 persona:
```
[System] I am a 34-year-old female living in an urban area of Argentina.
I completed university degree and currently work as a teacher.
My religious background is Catholic.

Answer the following survey question from my personal perspective...
```

**DeepPersona** — 풀 프로필:
```
[System] I am a 34-year-old female living in urban, Argentina.

Demographics:
- Education: university degree
- Marital status: married, has children
- Occupation: teacher (mid-career)

Personality:
- Openness: high
- Conscientiousness: high
- Extraversion: moderate
- Agreeableness: high
- Neuroticism: low

Values & beliefs:
- Core values: self-expression
- Life attitude: optimistic
- Religion: Catholic (moderately religious)
- Political leaning: center-left

(+ Social life, Lifestyle 등 추가 섹션)

Answer the following survey question from my personal perspective, genuinely
reflecting ALL aspects of my background, personality, values, and life
circumstances described above...
```

### 왜 이 3개를 비교하는가?

**"persona에 더 많은 정보를 줄수록 합성 데이터 품질이 좋아지는가?"**를 검증하기 위해서입니다.

### DeepPersona 논문 Table 2와의 관계

이 실험은 DeepPersona 논문의 Table 2 실험 설정을 **그대로 재현**합니다:
- 동일한 6개국, 동일한 WVS Wave 7 질문 6개, 동일한 3가지 prompting 전략 비교
- 다른 점: 이 프로젝트는 **GT-free 평가가 가능하다는 것을 추가로 보이는 것**

---

## 5. 3가지 GT-free 지표 (SCS, VCR, ICE)

### SCS (Synthetic Consistency Score) — "시험 답안의 일관성 검사"

**비유:** 6과목 시험에서 전부 만점이면 이상하고, 전부 0점이어도 이상하다. 건강한 학생이라면 과목별로 다소 차이가 있어야 한다.

```python
ALPHA_HEALTHY_CENTER = 0.7
def scs(df):
    alpha = cronbach_alpha(df)      # 6개 문항의 내적 일관성
    di_scs = abs(alpha - 0.7)       # 0.7에서 얼마나 벗어났나
```

- `DI_SCS = 0.0` → α가 정확히 0.7 → 매우 건강 (최선)
- `DI_SCS = 0.3` → α가 1.0 또는 0.4 → 이상 징후
- **낮을수록 좋음**

### VCR (Value Coherence Ratio) — "올원 찍기 탐지"

**비유:** "가족 중요도"와 "동성애 허용도"는 다른 가치 차원인데, LLM이 모든 질문에 일관되게 높은 값을 매긴다면 → halo effect.

```python
VCR_THRESHOLD = 0.5
def vcr(df):
    corr_matrix = items.corr().values
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    vcr_val = eigenvalues[0] / eigenvalues.sum()   # 1등 고유값 / 전체
    di_vcr = max(0.0, vcr_val - 0.5)
```

- `DI_VCR = 0.0` → 다차원적 응답 (정상)
- `DI_VCR = 0.3` → 심한 halo effect (결함)
- **낮을수록 좋음**

### ICE (Item Correlation Entropy) — "복사/붙여넣기 탐지"

**비유:** 15개 문항 쌍의 상관계수가 다양해야 실제 인간 같다. 전부 0.8 근처에 몰려있으면 복사한 느낌.

```python
def ice(df):
    corrs = [items_df.iloc[:,i].corr(items_df.iloc[:,j])
             for i,j in combinations(range(6), 2)]
    hist, _ = np.histogram(corrs, bins=20, range=(-1.0, 1.0))
    entropy = -sum(p * log2(p) for p in hist_normalized)
    di_ice = -normalized_entropy   # 부호 반전
```

- `DI_ICE = -0.9` → 매우 다양 (최선)
- `DI_ICE = -0.2` → 한 곳에 몰림 (결함)
- **더 음수일수록 좋음**

### 왜 GT 없이 작동하는가?

세 지표 모두 **합성 데이터 자체의 내부 구조만** 분석합니다. "건강한 설문 데이터라면 이런 통계적 성질을 가져야 한다"는 설문 측정론(psychometrics)의 원리를 활용한 것입니다.

---

## 6. 현재 진행 상황 (2026-03-29 기준)

### 완성된 것

- config, engine, prompts, metrics 모듈 전체
- WVS GT 데이터 (6개국 distributions.json + CSV)
- 파이프라인 스크립트 5개
- CLAUDE.md, README.md

### 미완성 항목 → **이번 세션에서 완료됨**

- Privacy GT 전처리: `data/preprocess_privacy.py` 생성
- main.py: 파이프라인 CLI 엔트리포인트로 교체
- vLLM 서버 시작 + 합성 데이터 생성: 실행 중

---

## 7. 기대하는 최종 결과

### "성공"의 기준

- Spearman ρ > 0.7, p < 0.01 → **강한 일치** → 핵심 주장 증명
- ρ = 0.4~0.7, p < 0.05 → **중간 일치** → 가능성 있음
- ρ < 0.4 또는 p > 0.05 → 주장 뒷받침 어려움

Per-country 분석에서 6개국 중 대부분(4개 이상)에서 3개 method의 순위가 일치하면 더 강력한 증거.

---

## 8. 핵심 한 줄 요약

> **"LLM이 만든 가짜 설문 데이터가 얼마나 엉터리인지, 진짜 인간 데이터 없이도 내부 통계 구조(SCS/VCR/ICE)만으로 진단할 수 있다."**
