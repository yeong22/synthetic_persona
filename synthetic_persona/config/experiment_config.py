"""
Experiment configuration — DeepPersona (Wang et al., NeurIPS 2025) 재현 + GT-free metrics.
Single Source of Truth: 모든 실험 설정이 이 파일에 집중.
"""
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Model (vLLM offline mode — no server, GPU freed on process exit)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
TEMPERATURE = 0.7
NUM_RESPONSES_PER_CONDITION = 300

# ---------------------------------------------------------------------------
# Prompting methods
# ---------------------------------------------------------------------------
PROMPTING_METHODS = ["cultural", "opencharacter", "deep_persona"]

# ---------------------------------------------------------------------------
# Domain: WVS Wave 7 — DeepPersona Table 2 / Appendix A.4
# ---------------------------------------------------------------------------
WVS_COUNTRIES = ["Argentina", "Australia", "Germany", "India", "Kenya", "United States"]

# DeepPersona Appendix A.4 문항 (WVS Q-codes)
WVS_ITEMS = {
    "Q45": {
        "code": "Q45",
        "text": "If greater respect for authority takes place in the near future, do you think it would be a good thing, a bad thing, or you don't mind?",
        "prompt_suffix": "If you think it would be a good thing, please reply 1. If you don't mind, please reply 2. If you think it would be a bad thing, please reply 3.",
        "scale": (1, 3),
        "labels": {1: "A good thing", 2: "Don't mind", 3: "A bad thing"},
        "label": "Respect for Authority",
    },
    "Q46": {
        "code": "Q46",
        "text": "Taking all things together, rate how happy you would say you are.",
        "prompt_suffix": "Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, 3 is Not very happy, 4 is Not at all happy.",
        "scale": (1, 4),
        "labels": {1: "Very happy", 2: "Quite happy", 3: "Not very happy", 4: "Not at all happy"},
        "label": "Feeling of Happiness",
    },
    "Q57": {
        "code": "Q57",
        "text": "Generally speaking, would you say that most people can be trusted or that you need to be very careful in dealing with people?",
        "prompt_suffix": "(1) Most people can be trusted. (2) Need to be very careful.",
        "scale": (1, 2),
        "labels": {1: "Most people can be trusted", 2: "Need to be very careful"},
        "label": "Trust on People",
    },
    "Q184": {
        "code": "Q184",
        "text": "How justifiable do you think abortion is?",
        "prompt_suffix": "Please indicate using a scale from 1 to 10, where 1 means never justifiable and 10 means always justifiable.",
        "scale": (1, 10),
        "labels": {1: "Never justifiable", 10: "Always justifiable"},
        "label": "Justifiability of Abortion",
    },
    "Q218": {
        "code": "Q218",
        "text": "Have you signed a petition?",
        "prompt_suffix": "(1) You have signed a petition. (2) You might do it. (3) You would never under any circumstances do it.",
        "scale": (1, 3),
        "labels": {1: "Have done", 2: "Might do", 3: "Would never do"},
        "label": "Petition Signing",
    },
    "Q254": {
        "code": "Q254",
        "text": "How proud are you to be your nationality?",
        "prompt_suffix": "1 means very proud, 2 means quite proud, 3 means not very proud, 4 means not at all proud.",
        "scale": (1, 4),
        "labels": {1: "Very proud", 2: "Quite proud", 3: "Not very proud", 4: "Not at all proud"},
        "label": "Pride of Nationality",
    },
}

# ---------------------------------------------------------------------------
# Domain: Big Five (IPIP-FFM) — DeepPersona Table 3
# ---------------------------------------------------------------------------
BF_COUNTRIES = ["Argentina", "Australia", "India"]

# IPIP 50-item Big Five (ipip.ori.org)
# keyed: "+" = positively keyed, "-" = reverse keyed (even-numbered items)
_BF_TEXTS = {
    "EXT1": ("I am the life of the party.", "+"),
    "EXT2": ("I don't talk a lot.", "-"),
    "EXT3": ("I feel comfortable around people.", "+"),
    "EXT4": ("I keep in the background.", "-"),
    "EXT5": ("I start conversations.", "+"),
    "EXT6": ("I have little to say.", "-"),
    "EXT7": ("I talk to a lot of different people at parties.", "+"),
    "EXT8": ("I don't like to draw attention to myself.", "-"),
    "EXT9": ("I don't mind being the center of attention.", "+"),
    "EXT10": ("I am quiet around strangers.", "-"),
    "EST1": ("I get stressed out easily.", "-"),
    "EST2": ("I am relaxed most of the time.", "+"),
    "EST3": ("I worry about things.", "-"),
    "EST4": ("I seldom feel blue.", "+"),
    "EST5": ("I am easily disturbed.", "-"),
    "EST6": ("I get upset easily.", "-"),
    "EST7": ("I change my mood a lot.", "-"),
    "EST8": ("I have frequent mood swings.", "-"),
    "EST9": ("I get irritated easily.", "-"),
    "EST10": ("I often feel blue.", "-"),
    "AGR1": ("I feel little concern for others.", "-"),
    "AGR2": ("I am interested in people.", "+"),
    "AGR3": ("I insult people.", "-"),
    "AGR4": ("I sympathize with others' feelings.", "+"),
    "AGR5": ("I am not interested in other people's problems.", "-"),
    "AGR6": ("I have a soft heart.", "+"),
    "AGR7": ("I am not really interested in others.", "-"),
    "AGR8": ("I take time out for others.", "+"),
    "AGR9": ("I feel others' emotions.", "+"),
    "AGR10": ("I make people feel at ease.", "+"),
    "CSN1": ("I am always prepared.", "+"),
    "CSN2": ("I leave my belongings around.", "-"),
    "CSN3": ("I pay attention to details.", "+"),
    "CSN4": ("I make a mess of things.", "-"),
    "CSN5": ("I get chores done right away.", "+"),
    "CSN6": ("I often forget to put things back in their proper place.", "-"),
    "CSN7": ("I like order.", "+"),
    "CSN8": ("I shirk my duties.", "-"),
    "CSN9": ("I follow a schedule.", "+"),
    "CSN10": ("I am exacting in my work.", "+"),
    "OPN1": ("I have a rich vocabulary.", "+"),
    "OPN2": ("I have difficulty understanding abstract ideas.", "-"),
    "OPN3": ("I have a vivid imagination.", "+"),
    "OPN4": ("I am not interested in abstract ideas.", "-"),
    "OPN5": ("I have excellent ideas.", "+"),
    "OPN6": ("I do not have a good imagination.", "-"),
    "OPN7": ("I am quick to understand things.", "+"),
    "OPN8": ("I use difficult words.", "+"),
    "OPN9": ("I spend time reflecting on things.", "+"),
    "OPN10": ("I am full of ideas.", "+"),
}

BF_FACTORS = {
    "EXT": [f"EXT{i}" for i in range(1, 11)],
    "EST": [f"EST{i}" for i in range(1, 11)],
    "AGR": [f"AGR{i}" for i in range(1, 11)],
    "CSN": [f"CSN{i}" for i in range(1, 11)],
    "OPN": [f"OPN{i}" for i in range(1, 11)],
}

BF_REVERSE_KEYED = {code for code, (_, keyed) in _BF_TEXTS.items() if keyed == "-"}

BF_ITEMS = {
    code: {
        "code": code,
        "text": text,
        "keyed": keyed,
        "factor": code[:3],
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    }
    for code, (text, keyed) in _BF_TEXTS.items()
}

# ---------------------------------------------------------------------------
# Domain: Privacy Calculus
# ---------------------------------------------------------------------------
PRIVACY_COUNTRIES = ["South Africa", "United Kingdom"]

PRIVACY_ITEMS = {
    "PC1_perceived_benefit": {
        "code": "PC1",
        "text": "Sharing my personal data with online services provides me with significant benefits.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC2_perceived_risk": {
        "code": "PC2",
        "text": "I am concerned that my personal data could be misused by online services.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC3_trust_provider": {
        "code": "PC3",
        "text": "I trust that online service providers will protect my personal information.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC4_info_sensitivity": {
        "code": "PC4",
        "text": "I consider my personal information to be highly sensitive.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC5_sharing_intention": {
        "code": "PC5",
        "text": "I am willing to share my personal data with online services in exchange for benefits.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC6_privacy_concern": {
        "code": "PC6",
        "text": "I am concerned about my privacy when using online services.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
}

# ---------------------------------------------------------------------------
# SCS healthy alpha center (도메인별)
# ---------------------------------------------------------------------------
ALPHA_HEALTHY_CENTER = {
    "wvs": 0.4,       # 다차원 가치 문항 → α가 원래 낮음
    "bigfive": 0.7,    # 요인 내 단일 구성개념
    "privacy": 0.7,    # 단일 구성개념
}

# ---------------------------------------------------------------------------
# GT data paths
# ---------------------------------------------------------------------------
WVS_GT_DIR = DATA_DIR / "wvs_gt"
BF_GT_DIR = DATA_DIR / "bigfive_gt"
PRIVACY_GT_DIR = DATA_DIR / "privacy_gt"

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCondition:
    domain: str
    country: str
    prompt_method: str
    items: dict = field(default_factory=dict)
    n_responses: int = NUM_RESPONSES_PER_CONDITION


def get_all_conditions(domain: str | None = None) -> list[ExperimentCondition]:
    """Generate experiment conditions. If domain is None, return all domains."""
    conditions = []

    domains = [domain] if domain else ["wvs", "bigfive", "privacy"]

    if "wvs" in domains:
        for country in WVS_COUNTRIES:
            for method in PROMPTING_METHODS:
                conditions.append(ExperimentCondition(
                    domain="wvs", country=country, prompt_method=method, items=WVS_ITEMS,
                ))

    if "bigfive" in domains:
        for country in BF_COUNTRIES:
            for method in PROMPTING_METHODS:
                conditions.append(ExperimentCondition(
                    domain="bigfive", country=country, prompt_method=method, items=BF_ITEMS,
                ))

    if "privacy" in domains:
        for country in PRIVACY_COUNTRIES:
            for method in PROMPTING_METHODS:
                conditions.append(ExperimentCondition(
                    domain="privacy", country=country, prompt_method=method, items=PRIVACY_ITEMS,
                ))

    return conditions
