"""
Experiment configuration for synthetic persona survey research.
DeepPersona Table 2의 WVS 실험과 동일한 설정.
"""
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
WVS_DIR = DATA_DIR / "wvs"
PRIVACY_DIR = DATA_DIR / "privacy_data"

# ---------------------------------------------------------------------------
# Model (vLLM offline mode — no server, GPU freed on process exit)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TENSOR_PARALLEL_SIZE = 2  # RTX 3090 × 2 (GPU 0,1); 3B model fits easily in 2 GPUs
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
TEMPERATURE = 0.7
NUM_RESPONSES_PER_CONDITION = 300  # synthetic 응답 수 per (country, prompt_method)

# ---------------------------------------------------------------------------
# Domain B: WVS Wave 7 — DeepPersona Table 2 설정
# ---------------------------------------------------------------------------
# 6개국 (DeepPersona 논문과 동일)
WVS_COUNTRIES = ["Argentina", "Australia", "Germany", "India", "Kenya", "United States"]
WVS_COUNTRY_CODES = {
    "Argentina": "ARG", "Australia": "AUS", "Germany": "DEU",
    "India": "IND", "Kenya": "KEN", "United States": "USA",
}

# WVS Wave 7 질문 6개 (DeepPersona Table 2와 동일)
# 각 질문의 원본 스케일을 그대로 사용
WVS_ITEMS = {
    "Q45": {
        "code": "Q45",
        "text": "For each of the following, indicate how important it is in your life: Family",
        "scale": (1, 3),
        "labels": {1: "Not very important", 2: "Rather important",
                   3: "Very important"},
    },
    "Q46": {
        "code": "Q46",
        "text": "For each of the following, indicate how important it is in your life: Friends",
        "scale": (1, 4),
        "labels": {1: "Very important", 2: "Rather important",
                   3: "Not very important", 4: "Not at all important"},
    },
    "Q57": {
        "code": "Q57",
        "text": "Generally speaking, would you say that most people can be trusted or that you need to be very careful in dealing with people?",
        "scale": (1, 2),
        "labels": {1: "Most people can be trusted",
                   2: "Need to be very careful"},
    },
    "Q184": {
        "code": "Q184",
        "text": "All things considered, how satisfied are you with your life as a whole these days?",
        "scale": (1, 10),
        "labels": {1: "Completely dissatisfied", 10: "Completely satisfied"},
    },
    "Q218": {
        "code": "Q218",
        "text": "Please tell me whether you think the following can always be justified, never be justified, or something in between: Homosexuality",
        "scale": (1, 3),
        "labels": {1: "Never justifiable", 2: "Something in between",
                   3: "Always justifiable"},
    },
    "Q254": {
        "code": "Q254",
        "text": "Here is a list of qualities that children can be encouraged to learn at home. How important is 'Tolerance and respect for other people'?",
        "scale": (1, 5),
        "labels": {1: "Very important", 2: "Important", 3: "Somewhat important",
                   4: "Not very important", 5: "Not important at all"},
    },
}

# ---------------------------------------------------------------------------
# Domain A: Privacy Calculus
# ---------------------------------------------------------------------------
PRIVACY_ITEMS = {
    "PC1_perceived_benefit": {
        "code": "PC1", "text": "Sharing my personal data with online services provides me with significant benefits.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC2_perceived_risk": {
        "code": "PC2", "text": "I am concerned that my personal data could be misused by online services.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC3_trust_provider": {
        "code": "PC3", "text": "I trust that online service providers will protect my personal information.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC4_info_sensitivity": {
        "code": "PC4", "text": "I consider my personal information to be highly sensitive.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC5_sharing_intention": {
        "code": "PC5", "text": "I am willing to share my personal data with online services in exchange for benefits.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC6_privacy_concern": {
        "code": "PC6", "text": "I am concerned about my privacy when using online services.",
        "scale": (1, 7), "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
}
PRIVACY_COUNTRIES = ["South Africa", "United Kingdom"]

# ---------------------------------------------------------------------------
# Prompting methods
# ---------------------------------------------------------------------------
PROMPTING_METHODS = ["cultural", "opencharacter", "deep_persona"]

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCondition:
    domain: str                  # "wvs" or "privacy"
    country: str
    prompt_method: str           # one of PROMPTING_METHODS
    items: dict = field(default_factory=dict)
    n_responses: int = NUM_RESPONSES_PER_CONDITION


def get_all_conditions() -> list[ExperimentCondition]:
    """Generate all experiment conditions for both domains."""
    conditions = []

    for country in WVS_COUNTRIES:
        for method in PROMPTING_METHODS:
            conditions.append(ExperimentCondition(
                domain="wvs", country=country, prompt_method=method, items=WVS_ITEMS,
            ))

    for country in PRIVACY_COUNTRIES:
        for method in PROMPTING_METHODS:
            conditions.append(ExperimentCondition(
                domain="privacy", country=country, prompt_method=method, items=PRIVACY_ITEMS,
            ))

    return conditions
