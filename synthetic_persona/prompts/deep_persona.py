"""
DeepPersona-style persona prompting.
참고: thzva/Deeppersona (GitHub)

핵심 아이디어:
  1) 7개 anchor attribute로 persona 뼈대 정의
  2) 200+ taxonomy attribute에서 점진적으로 세부 속성 샘플링
  3) LLM에게 anchor 기반으로 일관성 있는 세부 속성을 생성하도록 요청
     (progressive attribute sampling)

Qwen2.5-3B로 실행 가능하도록 단순화:
  - 전체 taxonomy를 하드코딩하여 LLM 의존 최소화
  - anchor → category → attribute 3단계 계층
  - 각 category에서 1-2개 attribute를 anchor와 일관되게 샘플링
"""
from __future__ import annotations
import random
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 7 Anchor Attributes
# ---------------------------------------------------------------------------
ANCHORS = {
    "age": {
        "desc": "Age group",
        "options": {
            "young_adult":  (18, 29),
            "adult":        (30, 44),
            "middle_aged":  (45, 59),
            "senior":       (60, 80),
        },
    },
    "gender": {
        "desc": "Gender",
        "options": ["male", "female"],
    },
    "location": {
        "desc": "Country and area type",
        # 국가별로 동적 설정
    },
    "career": {
        "desc": "Occupation and career stage",
        # taxonomy에서 상세 속성 파생
    },
    "values": {
        "desc": "Core value orientation",
        "options": [
            "traditional",      # 전통/보수적 가치
            "secular-rational",  # 세속적/합리적 가치
            "survival",         # 생존 지향
            "self-expression",  # 자기표현 지향
        ],
    },
    "life_attitude": {
        "desc": "General attitude toward life",
        "options": [
            "optimistic",  "pragmatic",  "cautious",
            "ambitious",   "content",    "anxious",
        ],
    },
    "interests": {
        "desc": "Primary interests and hobbies",
        # taxonomy에서 상세 속성 파생
    },
}


# ---------------------------------------------------------------------------
# Taxonomy: Category → Attributes (200+ total)
# anchor 값에 따라 일관성 있게 선택됨
# ---------------------------------------------------------------------------
TAXONOMY = {
    # === Demographic ===
    "marital_status": {
        "options": ["single", "married", "divorced", "widowed", "in a relationship"],
        "anchor_bias": {  # anchor value → 선호 옵션
            "young_adult": ["single", "in a relationship"],
            "senior": ["married", "widowed"],
        },
    },
    "children": {
        "options": ["no children", "one child", "two children", "three or more children"],
        "anchor_bias": {
            "young_adult": ["no children"],
            "senior": ["two children", "three or more children"],
        },
    },
    "income_level": {
        "options": ["low income", "lower-middle income", "middle income",
                    "upper-middle income", "high income"],
    },
    "education_level": {
        "options": ["primary school", "secondary school", "vocational training",
                    "bachelor's degree", "master's degree", "doctoral degree"],
    },

    # === Personality (Big Five) ===
    "openness": {
        "options": ["very low openness", "low openness", "moderate openness",
                    "high openness", "very high openness"],
        "anchor_bias": {
            "self-expression": ["high openness", "very high openness"],
            "traditional": ["low openness", "moderate openness"],
        },
    },
    "conscientiousness": {
        "options": ["very low", "low", "moderate", "high", "very high"],
    },
    "extraversion": {
        "options": ["very introverted", "introverted", "ambivert",
                    "extraverted", "very extraverted"],
        "anchor_bias": {
            "optimistic": ["extraverted", "very extraverted"],
            "anxious": ["introverted", "very introverted"],
        },
    },
    "agreeableness": {
        "options": ["very low", "low", "moderate", "high", "very high"],
    },
    "neuroticism": {
        "options": ["very low", "low", "moderate", "high", "very high"],
        "anchor_bias": {
            "anxious": ["high", "very high"],
            "content": ["very low", "low"],
        },
    },

    # === Social ===
    "social_circle": {
        "options": ["mostly family-oriented", "small close friend group",
                    "large diverse social network", "mostly colleagues",
                    "relatively isolated"],
    },
    "community_involvement": {
        "options": ["not involved", "occasionally volunteers",
                    "active community member", "community leader"],
    },
    "political_leaning": {
        "options": ["far left", "center-left", "centrist",
                    "center-right", "far right", "apolitical"],
        "anchor_bias": {
            "traditional": ["center-right", "far right"],
            "self-expression": ["center-left", "far left"],
            "secular-rational": ["centrist", "center-left"],
        },
    },

    # === Cultural ===
    "religion": {
        "options_by_country": {
            "Argentina": ["Catholic", "Evangelical", "non-religious", "other"],
            "Australia": ["Christian", "non-religious", "Buddhist", "Muslim", "other"],
            "Germany": ["Protestant", "Catholic", "non-religious", "Muslim", "other"],
            "India": ["Hindu", "Muslim", "Christian", "Sikh", "Buddhist", "other"],
            "Kenya": ["Protestant", "Catholic", "Muslim", "traditional", "non-religious"],
            "United States": ["Protestant", "Catholic", "non-religious", "Jewish", "other"],
        },
        "anchor_bias": {
            "traditional": None,      # 국가 내 최다 종교 선호 (별도 처리)
            "secular-rational": ["non-religious"],
        },
    },
    "religiosity": {
        "options": ["not religious at all", "slightly religious",
                    "moderately religious", "very religious", "deeply devout"],
        "anchor_bias": {
            "traditional": ["very religious", "deeply devout"],
            "secular-rational": ["not religious at all", "slightly religious"],
        },
    },
    "cultural_identity": {
        "options": ["strongly identifies with national culture",
                    "bicultural identity", "cosmopolitan outlook",
                    "primarily identifies with local/regional culture",
                    "identifies with a diaspora community"],
    },
    "media_consumption": {
        "options": ["mostly traditional media (TV, newspaper)",
                    "primarily social media", "balanced mix",
                    "mainly podcasts and online content", "minimal media consumption"],
    },

    # === Economic ===
    "financial_attitude": {
        "options": ["very frugal", "careful saver", "balanced spender",
                    "somewhat impulsive", "generous spender"],
        "anchor_bias": {
            "survival": ["very frugal", "careful saver"],
            "self-expression": ["balanced spender", "generous spender"],
        },
    },
    "housing": {
        "options": ["rents an apartment", "owns a house", "owns an apartment",
                    "lives with family", "shared housing"],
        "anchor_bias": {
            "young_adult": ["rents an apartment", "lives with family", "shared housing"],
            "senior": ["owns a house", "owns an apartment"],
        },
    },

    # === Health & Lifestyle ===
    "health_status": {
        "options": ["excellent health", "good health", "fair health",
                    "some chronic conditions", "poor health"],
    },
    "exercise_habits": {
        "options": ["sedentary", "light exercise weekly", "moderate exercise 3-4x/week",
                    "very active daily", "competitive athlete"],
    },
    "diet": {
        "options": ["no particular diet", "health-conscious",
                    "vegetarian", "traditional local diet", "restricted diet"],
    },

    # === Technology ===
    "tech_proficiency": {
        "options": ["very low (basic phone)", "low (basic internet)",
                    "moderate (regular smartphone user)", "high (early adopter)",
                    "very high (tech professional)"],
        "anchor_bias": {
            "young_adult": ["moderate (regular smartphone user)", "high (early adopter)"],
            "senior": ["very low (basic phone)", "low (basic internet)"],
        },
    },
    "social_media_usage": {
        "options": ["never uses", "rarely checks", "daily casual user",
                    "frequent active poster", "heavy user / influencer"],
    },

    # === Work ===
    "work_satisfaction": {
        "options": ["very dissatisfied", "somewhat dissatisfied", "neutral",
                    "satisfied", "very satisfied"],
    },
    "career_stage": {
        "options": ["student", "entry-level", "mid-career",
                    "senior professional", "retired"],
        "anchor_bias": {
            "young_adult": ["student", "entry-level"],
            "adult": ["mid-career"],
            "middle_aged": ["senior professional"],
            "senior": ["senior professional", "retired"],
        },
    },

    # === Interests (세부) ===
    "hobby_primary": {
        "options": [
            "reading", "sports", "cooking", "gardening", "music",
            "travel", "gaming", "crafts", "volunteering", "religious activities",
            "socializing", "watching TV/movies", "outdoor activities", "art",
        ],
    },
    "hobby_secondary": {
        "options": [
            "reading", "walking", "cooking", "music", "family time",
            "social media", "puzzles", "shopping", "fishing", "dancing",
        ],
    },
}

# 국가별 직업 풀
OCCUPATIONS_BY_COUNTRY = {
    "Argentina": [
        "teacher", "office worker", "shopkeeper", "farmer", "nurse",
        "engineer", "factory worker", "taxi driver", "accountant", "homemaker",
        "waiter", "mechanic", "government employee", "construction worker",
    ],
    "Australia": [
        "nurse", "teacher", "engineer", "tradesperson", "retail worker",
        "office worker", "IT professional", "farmer", "manager", "social worker",
        "mechanic", "chef", "police officer", "accountant",
    ],
    "Germany": [
        "engineer", "nurse", "teacher", "office clerk", "craftsperson",
        "IT specialist", "salesperson", "social worker", "factory worker",
        "manager", "mechanic", "researcher", "baker", "logistics worker",
    ],
    "India": [
        "farmer", "teacher", "IT professional", "shopkeeper", "office worker",
        "factory worker", "government clerk", "doctor", "engineer", "homemaker",
        "auto-rickshaw driver", "tailor", "construction worker", "bank employee",
    ],
    "Kenya": [
        "farmer", "teacher", "trader", "boda-boda driver", "nurse",
        "office worker", "small business owner", "construction worker",
        "IT worker", "homemaker", "security guard", "pastor", "mechanic",
    ],
    "United States": [
        "nurse", "teacher", "software developer", "retail worker", "office manager",
        "truck driver", "engineer", "lawyer", "construction worker", "homemaker",
        "cashier", "accountant", "firefighter", "police officer",
    ],
}

# 국가별 area 옵션
AREAS_BY_COUNTRY = {
    "Argentina": ["Buenos Aires metropolitan area", "urban Córdoba", "rural Pampas",
                   "suburban Mendoza", "Patagonia"],
    "Australia": ["Sydney metropolitan area", "Melbourne suburbs", "rural Queensland",
                   "Perth", "regional Victoria"],
    "Germany": ["Berlin", "Munich", "rural Bavaria", "suburban Hamburg",
                 "Ruhr industrial area", "small town in Saxony"],
    "India": ["Mumbai metropolitan area", "rural Uttar Pradesh", "Bangalore IT district",
               "Delhi NCR", "semi-urban Tamil Nadu", "rural Bihar"],
    "Kenya": ["Nairobi", "Mombasa", "rural Rift Valley", "Kisumu town",
               "peri-urban Kiambu", "pastoral Turkana"],
    "United States": ["New York metropolitan area", "suburban Texas", "rural Midwest",
                       "Los Angeles", "small town in Appalachia", "suburban Pacific Northwest"],
}


# ---------------------------------------------------------------------------
# Progressive Attribute Sampling
# ---------------------------------------------------------------------------
def _sample_anchors(country: str, rng: random.Random) -> dict:
    """7개 anchor attribute 샘플링."""
    # Age
    age_group = rng.choice(list(ANCHORS["age"]["options"].keys()))
    age_lo, age_hi = ANCHORS["age"]["options"][age_group]
    age = rng.randint(age_lo, age_hi)

    # Gender
    gender = rng.choice(ANCHORS["gender"]["options"])

    # Location
    areas = AREAS_BY_COUNTRY.get(country, ["urban area", "suburban area", "rural area"])
    area = rng.choice(areas)

    # Career
    occupations = OCCUPATIONS_BY_COUNTRY.get(country, ["office worker", "teacher"])
    career = rng.choice(occupations)

    # Values
    values = rng.choice(ANCHORS["values"]["options"])

    # Life attitude
    life_attitude = rng.choice(ANCHORS["life_attitude"]["options"])

    # Interests
    all_hobbies = TAXONOMY["hobby_primary"]["options"]
    interests = rng.sample(all_hobbies, min(2, len(all_hobbies)))

    return {
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "country": country,
        "area": area,
        "career": career,
        "values": values,
        "life_attitude": life_attitude,
        "interests": interests,
    }


def _sample_taxonomy_attr(
    category: str,
    anchors: dict,
    rng: random.Random,
) -> str:
    """
    anchor와 일관되게 taxonomy attribute를 샘플링 (progressive sampling).
    anchor_bias가 있으면 해당 옵션을 70% 확률로 선택.
    """
    cat_def = TAXONOMY[category]

    # 국가별 옵션이 있는 경우 (e.g., religion)
    if "options_by_country" in cat_def:
        country = anchors["country"]
        options = cat_def["options_by_country"].get(
            country, cat_def.get("options", ["other"])
        )
    else:
        options = cat_def["options"]

    biases = cat_def.get("anchor_bias", {})
    biased_options = None

    # anchor 값에 따른 bias 적용
    for anchor_key in ["age_group", "values", "life_attitude"]:
        anchor_val = anchors.get(anchor_key)
        if anchor_val in biases:
            bias = biases[anchor_val]
            if bias is None:
                # 특수 케이스: 국가 내 첫 번째 옵션 선호 (최다 종교 등)
                biased_options = options[:1]
            else:
                biased_options = [o for o in bias if o in options]
            break

    # 70% bias, 30% uniform
    if biased_options and rng.random() < 0.7:
        return rng.choice(biased_options)
    return rng.choice(options)


def _sample_full_persona(country: str, rng: random.Random) -> dict:
    """anchor + taxonomy의 전체 persona 속성 샘플링."""
    anchors = _sample_anchors(country, rng)

    # Progressive: anchor 기반으로 taxonomy 속성 순차 샘플링
    # 카테고리 선택 (전부 사용하면 프롬프트가 너무 길어지므로 핵심 카테고리만)
    SELECTED_CATEGORIES = [
        "marital_status", "children", "income_level", "education_level",
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
        "religion", "religiosity",
        "political_leaning", "social_circle", "community_involvement",
        "financial_attitude", "housing",
        "health_status", "exercise_habits",
        "tech_proficiency", "career_stage",
        "cultural_identity", "media_consumption",
    ]

    taxonomy_attrs = {}
    for cat in SELECTED_CATEGORIES:
        taxonomy_attrs[cat] = _sample_taxonomy_attr(cat, anchors, rng)

    return {**anchors, "taxonomy": taxonomy_attrs}


# ---------------------------------------------------------------------------
# Persona → System Prompt 변환
# ---------------------------------------------------------------------------
def _persona_to_prompt(p: dict) -> str:
    """전체 persona dict를 자연어 시스템 프롬프트로 변환."""
    tax = p["taxonomy"]
    interests_str = " and ".join(p["interests"])

    # 구조화된 persona 프로필
    profile = f"""I am a {p['age']}-year-old {p['gender']} living in {p['area']}, {p['country']}.

Demographics:
- Education: {tax['education_level']}
- Marital status: {tax['marital_status']}, {tax['children']}
- Occupation: {p['career']} ({tax['career_stage']})
- Income: {tax['income_level']}
- Housing: {tax['housing']}

Personality:
- Openness: {tax['openness']}
- Conscientiousness: {tax['conscientiousness']}
- Extraversion: {tax['extraversion']}
- Agreeableness: {tax['agreeableness']}
- Neuroticism: {tax['neuroticism']}

Values & beliefs:
- Core values: {p['values']}
- Life attitude: {p['life_attitude']}
- Religion: {tax['religion']} ({tax['religiosity']})
- Political leaning: {tax['political_leaning']}
- Cultural identity: {tax['cultural_identity']}

Social life:
- Social circle: {tax['social_circle']}
- Community: {tax['community_involvement']}
- Media: {tax['media_consumption']}

Lifestyle:
- Health: {tax['health_status']}
- Exercise: {tax['exercise_habits']}
- Tech proficiency: {tax['tech_proficiency']}
- Financial attitude: {tax['financial_attitude']}
- Interests: {interests_str}"""

    return (
        f"{profile}\n\n"
        f"Answer the following survey question from my personal perspective, "
        f"genuinely reflecting ALL aspects of my background, personality, values, and life circumstances described above. "
        f"Respond with ONLY a single integer number on the given scale. "
        f"Do not add any explanation or reasoning."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_persona(
    country: str,
    n: int,
    seed: int = 42,
    client=None,
    use_llm_enrichment: bool = False,
    **kwargs,
) -> list[str]:
    """
    n개의 DeepPersona-style 시스템 프롬프트 생성.

    Args:
        country: 대상 국가
        n: 생성할 persona 수
        seed: 랜덤 시드
        client: LLMClient (use_llm_enrichment=True일 때 필요)
        use_llm_enrichment: True이면 LLM으로 추가 속성 보강 (선택적)

    Returns:
        list[str]: n개의 시스템 프롬프트
    """
    rng = random.Random(seed)
    personas = []

    for i in range(n):
        p = _sample_full_persona(country, rng)

        # (선택) LLM enrichment: anchor 기반으로 추가 서술 생성
        if use_llm_enrichment and client is not None:
            p = _llm_enrich_persona(p, client)

        prompt = _persona_to_prompt(p)
        personas.append(prompt)

    return personas


def _llm_enrich_persona(persona: dict, client) -> dict:
    """
    LLM을 사용하여 persona에 일관된 backstory 추가 (선택적).
    Qwen2.5-3B 기준 짧은 프롬프트로 제한.
    """
    tax = persona["taxonomy"]
    summary = (
        f"{persona['age']}-year-old {persona['gender']} {persona['career']} "
        f"in {persona['area']}, {persona['country']}. "
        f"Values: {persona['values']}. Attitude: {persona['life_attitude']}. "
        f"Religion: {tax['religion']}."
    )

    enrichment_prompt = (
        f"Given this person: {summary}\n\n"
        f"Write ONE sentence about a specific life experience that shaped their worldview. "
        f"Be concrete and specific. Only output the sentence, nothing else."
    )

    try:
        backstory = client.query(
            system_prompt="You create brief, realistic character backstories.",
            user_prompt=enrichment_prompt,
            max_tokens=80,
        )
        persona["backstory"] = backstory.strip()
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}")
        persona["backstory"] = ""

    return persona


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    countries = ["Argentina", "Australia", "Germany", "India", "Kenya", "United States"]

    for c in countries:
        personas = generate_persona(c, 2, seed=0)
        print(f"\n{'='*70}")
        print(f"Country: {c}")
        for i, p in enumerate(personas):
            print(f"\n--- Persona {i} ---")
            # 프로필 부분만 출력 (마지막 instruction 제외)
            profile_end = p.index("\n\nAnswer the following")
            print(p[:profile_end])
        print(f"\nPrompt length: {len(personas[0])} chars")

    # 통계 출력
    print(f"\n{'='*70}")
    print("Taxonomy categories:", len(TAXONOMY))
    total_options = sum(len(v.get("options", v.get("options_by_country", {}).get("United States", [])))
                        for v in TAXONOMY.values())
    print(f"Total attribute options: ~{total_options}")
