# 프로젝트 진행 상황 리포트
## 작성일시: 2026-03-30 13:48 KST (최종 업데이트)

---

### 1. 전체 파이프라인 진행률

- [x] Phase 0: 환경 세팅 — **완료**
- [x] Phase 1: 프로젝트 구조 생성 — **완료**
- [x] Phase 2: vLLM 오프라인 모드 전환 — **완료** (서버 방식 폐기, GPU 자동 해제)
- [x] Phase 3: Persona 시스템 프롬프트 — **완료** (Cultural에 age/gender variation 추가)
- [x] Phase 4: 메트릭 계산 코드 — **완료** (Step A/B/C + Analysis, Privacy 지원 추가)
- [x] Phase 5: Domain B (WVS) 실험 + 메트릭 + 분석 — **완료**
- [x] Phase 6: Domain A (Privacy) 실험 + 메트릭 — **완료**

---

### 2. 실험 설정 변경사항

| 항목 | 이전 | 현재 | 이유 |
|------|------|------|------|
| vLLM | 서버 방식 (HTTP) | 오프라인 모드 (LLM 클래스) | 세션 끊김 시 GPU 좀비 방지 |
| temperature | 0.0 | 0.7 | cultural prompting 분산 0 문제 해결 |
| cultural prompting | 동일 프롬프트 x 300 | age/gender variation x 300 | 응답 다양성 확보 |
| Step A | WVS만 | WVS + Privacy | Privacy 도메인 추가 |
| VCR 계산 | zero-variance item 포함 → NaN | zero-variance item 제외 | NaN 방지 |

---

### 3. 실험 결과 요약

#### GT-based 메트릭 (Step A) — 순위 패턴

**WVS 6개국 전체**: cultural > opencharacter > deep_persona (WD 기준, 높을수록 GT에서 멂)
→ **DeepPersona가 가장 GT에 근접**, cultural이 가장 멂

**Privacy 2개국**: South Africa에서 deep_persona의 WD가 가장 높음 (역전) — UK GT 데이터가 매우 적어(~17명) 신뢰도 낮음

#### GT-free 메트릭 (Step C) — DI 지표

| 지표 | WD와의 Spearman ρ | 해석 |
|------|-------------------|------|
| **DI_ICE** | **+0.558** (p<0.01) | ICE가 GT 순위를 가장 잘 예측 |
| DI_SCS | -0.768 (p<0.001) | alpha 기반 — 역방향 상관 (해석 주의) |
| DI_VCR | N/A (상수) | 모든 조건에서 VCR < 0.5 → DI_VCR=0 |
| DI_combined | +0.355 (n.s.) | SCS와 ICE의 상충으로 약화됨 |

#### JSD 기준 (정보이론적으로 더 적절한 GT 메트릭):

| 지표 | JSD와의 Spearman ρ | p-value |
|------|-------------------|---------|
| **DI_combined** | **+0.655** | p<0.001 |
| DI_ICE | +0.797 | p<0.001 |
| DI_SCS | -0.528 | p<0.01 |

→ **JSD 기준으로는 DI_combined이 유의미한 양의 상관** (ρ=0.655***)

#### Per-country DI_combined 순위 일치율

| 국가 | WD rank == DI_combined rank? |
|------|------------------------------|
| Argentina | NO |
| Australia | YES |
| Germany | YES |
| India | YES |
| Kenya | YES |
| United States | YES |
| South Africa | NO |
| United Kingdom | YES |
| **일치율** | **6/8 (75%)** |

#### Structural Fidelity (Step B)

SFS 순위도 WVS 6개국에서 동일 패턴: cultural < opencharacter < deep_persona
→ DeepPersona가 GT의 상관 구조를 가장 잘 재현

---

### 4. 파일 시스템 현황

```
results/
├── wvs/
│   ├── Argentina/       ← cultural.csv, opencharacter.csv, deep_persona.csv (각 300건)
│   ├── Australia/       ← 〃
│   ├── Germany/         ← 〃
│   ├── India/           ← 〃
│   ├── Kenya/           ← 〃
│   └── United States/   ← 〃
├── privacy/
│   ├── South Africa/    ← cultural.csv, opencharacter.csv, deep_persona.csv (각 300건)
│   └── United Kingdom/  ← 〃
└── metrics/
    ├── step_a_results.json    ← 24 conditions, GT-based (WD/JSD/KS/MeanDiff)
    ├── step_b_results.json    ← 24 conditions, Structural (SignF/SigF/NullF/SFS)
    ├── step_c_results.json    ← 24 conditions, GT-free (SCS/VCR/ICE/DI_combined)
    └── analysis_results.json  ← Spearman/Kendall concordance analysis
```

---

### 5. 알려진 제한사항

1. **DI_SCS 역방향 상관**: Cronbach's alpha 기반 DI_SCS가 GT 메트릭과 음의 상관. Cultural prompting은 낮은 alpha(~0.07-0.08)를 보여 DI_SCS가 낮지만, GT에서는 가장 나쁜 성능. DI_SCS의 부호 또는 가중치 재검토 필요.

2. **DI_VCR 상수**: 모든 24 조건에서 VCR < 0.5 → DI_VCR = 0. Threshold(0.5)가 너무 높거나, Qwen 3B 모델이 halo effect를 생성하지 않는 것으로 보임. Threshold 조정 또는 DI_combined에서 VCR 제외 검토.

3. **Privacy GT 데이터 부족**: UK는 ~17명으로 GT 자체의 신뢰도가 낮음. South Africa에서 deep_persona의 WD가 역전(가장 높음) — Privacy 도메인에서 deep_persona가 WVS에 특화된 편향일 가능성.

4. **WVS zero-variance items**: Q45(1-3), Q57(1-2) 등 짧은 스케일의 item은 temperature=0.7에서도 cultural에서 분산이 0. VCR 계산에서 자동 제외되도록 코드 수정 완료.

---

### 6. 핵심 발견 (논문용)

> **ICE (Item Correlation Entropy)가 가장 유망한 GT-free 지표.**
> JSD와 ρ=+0.797(p<0.001), WD와 ρ=+0.558(p<0.01).
> GT 없이도 합성 데이터의 품질 순위를 유의미하게 예측할 수 있음.

> **DI_combined은 JSD 기준으로 유의미** (ρ=+0.655, p<0.001).
> 단, WD 기준으로는 SCS의 역방향 효과로 상쇄되어 유의미하지 않음.
> → SCS의 가중치 조정 또는 ICE 단독 지표 사용 검토.
