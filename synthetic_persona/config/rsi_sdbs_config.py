"""
RSI (Response Stability Index) & SDBS (Social Desirability Bias Score) 설정.

RSI: paraphrase 쌍(동의어 재표현) + reverse 쌍(역방향) 정의.
SDBS: 각 문항의 사회적 바람직성 방향(SD+/SD-) 정의.

Big Five: reverse 쌍은 이미 내장 (홀수=+, 짝수=-). paraphrase만 추가 정의.
WVS: 역문항 없으므로 paraphrase만 정의. reverse는 별도 생성.
Privacy: paraphrase + 기존 construct 내 reverse 관계 활용.
"""

# ============================================================================
# WVS — DeepPersona A.4 문항의 paraphrase, reverse, SD direction
# ============================================================================
WVS_PARAPHRASE = {
    # code: (원본 text, paraphrase text, same scale/prompt_suffix)
    "Q45": {
        "code": "Q45_para",
        "text": "Do you think it would be good or bad if people in the future showed more respect for those in positions of authority?",
        "prompt_suffix": "If you think it would be a good thing, please reply 1. If you don't mind, please reply 2. If you think it would be a bad thing, please reply 3.",
        "scale": (1, 3),
        "labels": {1: "A good thing", 2: "Don't mind", 3: "A bad thing"},
    },
    "Q46": {
        "code": "Q46_para",
        "text": "Overall, how would you describe your level of happiness in life?",
        "prompt_suffix": "Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, 3 is Not very happy, 4 is Not at all happy.",
        "scale": (1, 4),
        "labels": {1: "Very happy", 2: "Quite happy", 3: "Not very happy", 4: "Not at all happy"},
    },
    "Q57": {
        "code": "Q57_para",
        "text": "In general, do you believe that the majority of people are trustworthy, or should you exercise caution when interacting with others?",
        "prompt_suffix": "(1) Most people can be trusted. (2) Need to be very careful.",
        "scale": (1, 2),
        "labels": {1: "Most people can be trusted", 2: "Need to be very careful"},
    },
    "Q184": {
        "code": "Q184_para",
        "text": "To what extent do you think terminating a pregnancy can be morally justified?",
        "prompt_suffix": "Please indicate using a scale from 1 to 10, where 1 means never justifiable and 10 means always justifiable.",
        "scale": (1, 10),
        "labels": {1: "Never justifiable", 10: "Always justifiable"},
    },
    "Q218": {
        "code": "Q218_para",
        "text": "Have you ever put your name on a formal written request or appeal addressed to authorities?",
        "prompt_suffix": "(1) You have done it. (2) You might do it. (3) You would never under any circumstances do it.",
        "scale": (1, 3),
        "labels": {1: "Have done", 2: "Might do", 3: "Would never do"},
    },
    "Q254": {
        "code": "Q254_para",
        "text": "How much national pride do you feel about being a citizen of your country?",
        "prompt_suffix": "1 means very proud, 2 means quite proud, 3 means not very proud, 4 means not at all proud.",
        "scale": (1, 4),
        "labels": {1: "Very proud", 2: "Quite proud", 3: "Not very proud", 4: "Not at all proud"},
    },
}

# WVS reverse pairs: 원본의 의미를 반전한 문항
WVS_REVERSE = {
    "Q45": {
        "code": "Q45_rev",
        "text": "If respect for authority were to decline in the near future, do you think it would be a good thing, a bad thing, or you don't mind?",
        "prompt_suffix": "If you think it would be a good thing, please reply 1. If you don't mind, please reply 2. If you think it would be a bad thing, please reply 3.",
        "scale": (1, 3),
        "labels": {1: "A good thing", 2: "Don't mind", 3: "A bad thing"},
    },
    "Q46": {
        "code": "Q46_rev",
        "text": "Taking all things together, rate how unhappy you would say you are.",
        "prompt_suffix": "Please use a scale from 1 to 4, where 1 is Very unhappy, 2 is Quite unhappy, 3 is Not very unhappy, 4 is Not at all unhappy.",
        "scale": (1, 4),
        "labels": {1: "Very unhappy", 2: "Quite unhappy", 3: "Not very unhappy", 4: "Not at all unhappy"},
    },
    "Q57": {
        "code": "Q57_rev",
        "text": "Generally speaking, would you say that most people cannot be trusted and you always need to be very careful?",
        "prompt_suffix": "(1) Most people cannot be trusted. (2) Most people can actually be trusted.",
        "scale": (1, 2),
        "labels": {1: "Most people cannot be trusted", 2: "Most people can be trusted"},
    },
    "Q184": {
        "code": "Q184_rev",
        "text": "How unjustifiable do you think abortion is?",
        "prompt_suffix": "Please indicate using a scale from 1 to 10, where 1 means completely justifiable and 10 means never justifiable.",
        "scale": (1, 10),
        "labels": {1: "Completely justifiable", 10: "Never justifiable"},
    },
    "Q218": {
        "code": "Q218_rev",
        "text": "Would you refuse to sign a petition under any circumstances?",
        "prompt_suffix": "(1) You would always refuse. (2) You might consider it. (3) You have no problem signing petitions.",
        "scale": (1, 3),
        "labels": {1: "Always refuse", 2: "Might consider", 3: "No problem"},
    },
    "Q254": {
        "code": "Q254_rev",
        "text": "How ashamed or indifferent do you feel about your nationality?",
        "prompt_suffix": "1 means very ashamed, 2 means somewhat ashamed, 3 means indifferent, 4 means not ashamed at all.",
        "scale": (1, 4),
        "labels": {1: "Very ashamed", 2: "Somewhat ashamed", 3: "Indifferent", 4: "Not ashamed"},
    },
}

# WVS SD direction: "+" = socially desirable = high value, "-" = SD = low value
WVS_SD_DIRECTION = {
    "Q45": "+",   # Respect for authority → socially desirable to support
    "Q46": "-",   # Happiness: low = very happy (SD+), so direction is -
    "Q57": "-",   # Trust: 1=trusting (SD+), 2=careful → direction is -
    "Q184": None, # Abortion: no clear SD direction (culture-dependent)
    "Q218": "-",  # Petition: 1=have done (civic engagement, SD+) → direction is -
    "Q254": "-",  # Pride: 1=very proud (SD+) → direction is -
}

# ============================================================================
# Big Five — paraphrase 쌍 (요인별 대표 2쌍씩 = 10쌍)
# reverse 쌍은 이미 내장 (홀수=+, 짝수=-)
# ============================================================================
BF_PARAPHRASE = {
    "EXT1": {
        "code": "EXT1_para",
        "text": "I am usually the most energetic and social person at gatherings.",
        "factor": "EXT", "keyed": "+",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "EXT2": {
        "code": "EXT2_para",
        "text": "I tend to be quiet and don't speak up much in conversations.",
        "factor": "EXT", "keyed": "-",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "EST1": {
        "code": "EST1_para",
        "text": "I become anxious and tense easily under pressure.",
        "factor": "EST", "keyed": "-",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "EST2": {
        "code": "EST2_para",
        "text": "I generally feel calm and at ease most of the time.",
        "factor": "EST", "keyed": "+",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "AGR1": {
        "code": "AGR1_para",
        "text": "I don't really care much about how other people feel.",
        "factor": "AGR", "keyed": "-",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "AGR2": {
        "code": "AGR2_para",
        "text": "I find other people fascinating and enjoy getting to know them.",
        "factor": "AGR", "keyed": "+",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "CSN1": {
        "code": "CSN1_para",
        "text": "I make sure to plan ahead and be ready for things.",
        "factor": "CSN", "keyed": "+",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "CSN2": {
        "code": "CSN2_para",
        "text": "I tend to be messy and leave things scattered around.",
        "factor": "CSN", "keyed": "-",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "OPN1": {
        "code": "OPN1_para",
        "text": "I know a lot of words and can express myself eloquently.",
        "factor": "OPN", "keyed": "+",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
    "OPN2": {
        "code": "OPN2_para",
        "text": "I struggle to grasp abstract or theoretical concepts.",
        "factor": "OPN", "keyed": "-",
        "scale": (1, 5),
        "labels": {1: "Very Inaccurate", 2: "Moderately Inaccurate",
                   3: "Neither Accurate Nor Inaccurate",
                   4: "Moderately Accurate", 5: "Very Accurate"},
    },
}

# Big Five reverse pairs: built-in (odd=+, even=-) within each factor
# e.g., EXT1(+) ↔ EXT2(-), EXT3(+) ↔ EXT4(-), etc.
BF_REVERSE_PAIRS = [
    ("EXT1", "EXT2"), ("EXT3", "EXT4"), ("EXT5", "EXT6"), ("EXT7", "EXT8"), ("EXT9", "EXT10"),
    ("EST2", "EST1"), ("EST4", "EST3"),  # EST: 2(+) vs 1(-), 4(+) vs 3(-)
    ("AGR2", "AGR1"), ("AGR4", "AGR3"), ("AGR6", "AGR5"), ("AGR8", "AGR7"),
    ("CSN1", "CSN2"), ("CSN3", "CSN4"), ("CSN5", "CSN6"), ("CSN7", "CSN8"),
    ("OPN1", "OPN2"), ("OPN3", "OPN4"), ("OPN5", "OPN6"), ("OPN7", "OPN8"),
]

# Big Five SD direction: socially desirable traits
BF_SD_DIRECTION = {
    "EXT": "+",   # Extraversion: socially desirable
    "EST": "+",   # Emotional Stability: socially desirable
    "AGR": "+",   # Agreeableness: socially desirable
    "CSN": "+",   # Conscientiousness: socially desirable
    "OPN": "+",   # Openness: socially desirable
}

# ============================================================================
# Privacy Calculus — paraphrase, reverse, SD direction
# ============================================================================
PRIVACY_PARAPHRASE = {
    "PC1": {
        "code": "PC1_para",
        "text": "I receive meaningful advantages from providing my personal information to online platforms.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC2": {
        "code": "PC2_para",
        "text": "I worry that online platforms might mishandle or exploit my personal data.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC3": {
        "code": "PC3_para",
        "text": "I have confidence that online service companies will safeguard my personal data.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC4": {
        "code": "PC4_para",
        "text": "I regard my personal information as very private and sensitive.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC5": {
        "code": "PC5_para",
        "text": "I am open to providing my personal data to online services if I get something valuable in return.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
    "PC6": {
        "code": "PC6_para",
        "text": "I have worries about my privacy being compromised when I use internet-based services.",
        "scale": (1, 7),
        "labels": {1: "Strongly disagree", 7: "Strongly agree"},
    },
}

# Privacy reverse pairs: conceptual inverses
PRIVACY_REVERSE_PAIRS = [
    ("PC1", "PC2"),  # benefit vs risk
    ("PC3", "PC2"),  # trust vs risk
    ("PC5", "PC6"),  # sharing intention vs privacy concern
]

# Privacy SD direction
PRIVACY_SD_DIRECTION = {
    "PC1": "+",   # Seeing benefits = positive/optimistic (SD+)
    "PC2": "+",   # Concern about misuse = responsible/cautious (SD+)
    "PC3": "+",   # Trusting providers = positive outlook (SD+)
    "PC4": "+",   # Valuing privacy = responsible (SD+)
    "PC5": "-",   # Willing to share = possibly naive (SD-)
    "PC6": "+",   # Privacy concern = responsible (SD+)
}
