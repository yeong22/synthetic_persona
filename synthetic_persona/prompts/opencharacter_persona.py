"""
OpenCharacter-style persona prompting.
GPT/LLM으로 생성한 것과 유사한 짧은 persona bio (3-5문장).
국가별 인구통계 분포에서 속성을 샘플링한 뒤 자연어 bio로 조합.
Anthology의 Bio 방식과 유사.
"""
from __future__ import annotations
import random

# ---------------------------------------------------------------------------
# 국가별 인구통계 분포 (Census/WVS 기반 간소화)
# ---------------------------------------------------------------------------
DEMOGRAPHICS = {
    "Argentina": {
        "age_dist": [(18, 29, 0.25), (30, 44, 0.30), (45, 59, 0.25), (60, 80, 0.20)],
        "gender": [("male", 0.48), ("female", 0.52)],
        "education": [
            ("primary school", 0.15), ("secondary school", 0.40),
            ("university degree", 0.35), ("postgraduate degree", 0.10),
        ],
        "occupation": [
            "teacher", "office worker", "shopkeeper", "farmer", "nurse",
            "engineer", "factory worker", "taxi driver", "accountant", "homemaker",
        ],
        "religion": [
            ("Catholic", 0.55), ("Evangelical Christian", 0.15),
            ("non-religious", 0.25), ("other", 0.05),
        ],
        "area": [("urban", 0.70), ("suburban", 0.15), ("rural", 0.15)],
    },
    "Australia": {
        "age_dist": [(18, 29, 0.20), (30, 44, 0.28), (45, 59, 0.27), (60, 80, 0.25)],
        "gender": [("male", 0.49), ("female", 0.51)],
        "education": [
            ("secondary school", 0.30), ("vocational diploma", 0.25),
            ("bachelor's degree", 0.30), ("postgraduate degree", 0.15),
        ],
        "occupation": [
            "nurse", "teacher", "engineer", "tradesperson", "retail worker",
            "office worker", "IT professional", "farmer", "manager", "social worker",
        ],
        "religion": [
            ("Christian", 0.40), ("non-religious", 0.45),
            ("Buddhist", 0.05), ("Muslim", 0.03), ("other", 0.07),
        ],
        "area": [("urban", 0.65), ("suburban", 0.25), ("rural", 0.10)],
    },
    "Germany": {
        "age_dist": [(18, 29, 0.17), (30, 44, 0.25), (45, 59, 0.28), (60, 80, 0.30)],
        "gender": [("male", 0.49), ("female", 0.51)],
        "education": [
            ("Hauptschule/Realschule", 0.30), ("Abitur", 0.25),
            ("university degree", 0.30), ("postgraduate degree", 0.15),
        ],
        "occupation": [
            "engineer", "nurse", "teacher", "office clerk", "craftsperson",
            "IT specialist", "salesperson", "social worker", "factory worker", "manager",
        ],
        "religion": [
            ("Protestant Christian", 0.25), ("Catholic", 0.25),
            ("non-religious", 0.40), ("Muslim", 0.06), ("other", 0.04),
        ],
        "area": [("urban", 0.55), ("suburban", 0.30), ("rural", 0.15)],
    },
    "India": {
        "age_dist": [(18, 29, 0.35), (30, 44, 0.30), (45, 59, 0.22), (60, 80, 0.13)],
        "gender": [("male", 0.52), ("female", 0.48)],
        "education": [
            ("primary school", 0.20), ("secondary school", 0.30),
            ("bachelor's degree", 0.35), ("postgraduate degree", 0.15),
        ],
        "occupation": [
            "farmer", "teacher", "IT professional", "shopkeeper", "office worker",
            "factory worker", "government clerk", "doctor", "engineer", "homemaker",
        ],
        "religion": [
            ("Hindu", 0.80), ("Muslim", 0.14),
            ("Christian", 0.02), ("Sikh", 0.02), ("other", 0.02),
        ],
        "area": [("urban", 0.35), ("semi-urban", 0.25), ("rural", 0.40)],
    },
    "Kenya": {
        "age_dist": [(18, 29, 0.40), (30, 44, 0.30), (45, 59, 0.18), (60, 80, 0.12)],
        "gender": [("male", 0.50), ("female", 0.50)],
        "education": [
            ("primary school", 0.30), ("secondary school", 0.35),
            ("university degree", 0.25), ("postgraduate degree", 0.10),
        ],
        "occupation": [
            "farmer", "teacher", "trader", "boda-boda driver", "nurse",
            "office worker", "small business owner", "construction worker", "IT worker", "homemaker",
        ],
        "religion": [
            ("Protestant Christian", 0.45), ("Catholic", 0.20),
            ("Muslim", 0.10), ("traditional/other", 0.10), ("non-religious", 0.15),
        ],
        "area": [("urban", 0.30), ("peri-urban", 0.20), ("rural", 0.50)],
    },
    "United States": {
        "age_dist": [(18, 29, 0.20), (30, 44, 0.26), (45, 59, 0.26), (60, 80, 0.28)],
        "gender": [("male", 0.49), ("female", 0.51)],
        "education": [
            ("high school diploma", 0.27), ("some college", 0.20),
            ("bachelor's degree", 0.33), ("postgraduate degree", 0.20),
        ],
        "occupation": [
            "nurse", "teacher", "software developer", "retail worker", "office manager",
            "truck driver", "engineer", "lawyer", "construction worker", "homemaker",
        ],
        "religion": [
            ("Protestant Christian", 0.40), ("Catholic", 0.20),
            ("non-religious", 0.28), ("Jewish", 0.02), ("other", 0.10),
        ],
        "area": [("urban", 0.55), ("suburban", 0.30), ("rural", 0.15)],
    },
}

_DEFAULT_DEMOGRAPHICS = DEMOGRAPHICS["United States"]


# ---------------------------------------------------------------------------
# 샘플링 유틸리티
# ---------------------------------------------------------------------------
def _weighted_choice(rng: random.Random, options: list[tuple]) -> str:
    """(value, weight) 리스트에서 가중치 기반 랜덤 선택."""
    values, weights = zip(*options)
    return rng.choices(values, weights=weights, k=1)[0]


def _sample_age(rng: random.Random, age_dist: list[tuple]) -> int:
    """(lo, hi, weight) 분포에서 나이 샘플링."""
    bucket = rng.choices(age_dist, weights=[d[2] for d in age_dist], k=1)[0]
    return rng.randint(bucket[0], bucket[1])


def _sample_demographics(country: str, rng: random.Random) -> dict:
    """국가별 인구통계 속성 샘플링."""
    demo = DEMOGRAPHICS.get(country, _DEFAULT_DEMOGRAPHICS)
    return {
        "country": country,
        "age": _sample_age(rng, demo["age_dist"]),
        "gender": _weighted_choice(rng, demo["gender"]),
        "education": _weighted_choice(rng, demo["education"]),
        "occupation": rng.choice(demo["occupation"]),
        "religion": _weighted_choice(rng, demo["religion"]),
        "area": _weighted_choice(rng, demo["area"]),
    }


# ---------------------------------------------------------------------------
# Bio 생성
# ---------------------------------------------------------------------------
def _build_bio(attrs: dict) -> str:
    """샘플링된 속성으로 3-5문장 자연어 bio 생성."""
    # 성별 대명사
    pronoun = "He" if attrs["gender"] == "male" else "She"
    article = "an" if attrs["area"][0].lower() in "aeiou" else "a"

    lines = [
        f"I am a {attrs['age']}-year-old {attrs['gender']} living in {article} "
        f"{attrs['area']} area of {attrs['country']}.",

        f"I completed {attrs['education']} and currently work as a {attrs['occupation']}.",

        f"My religious background is {attrs['religion']}.",
    ]
    return " ".join(lines)


def _build_system_prompt(bio: str) -> str:
    """Bio를 포함한 시스템 프롬프트 조립."""
    return (
        f"{bio}\n\n"
        f"Answer the following survey question from my personal perspective, "
        f"reflecting my background and life experience. "
        f"Respond with ONLY a single integer number on the given scale. "
        f"Do not add any explanation or reasoning."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_persona(country: str, n: int, seed: int = 42, **kwargs) -> list[str]:
    """
    n개의 OpenCharacter-style persona 시스템 프롬프트 생성.
    각 persona는 국가별 인구통계 분포에서 독립적으로 샘플링.
    """
    rng = random.Random(seed)
    personas = []
    for _ in range(n):
        attrs = _sample_demographics(country, rng)
        bio = _build_bio(attrs)
        personas.append(_build_system_prompt(bio))
    return personas


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    countries = ["Argentina", "Australia", "Germany", "India", "Kenya", "United States"]
    for c in countries:
        personas = generate_persona(c, 3, seed=0)
        print(f"\n{'='*60}")
        print(f"Country: {c} ({len(personas)} personas)")
        for i, p in enumerate(personas):
            # bio 부분만 출력
            bio_end = p.index("\n\n")
            print(f"  [{i}] {p[:bio_end]}")
