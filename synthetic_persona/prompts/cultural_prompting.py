"""
Cultural Prompting (Tao et al. 2024).
국적 + 최소한의 인구통계(나이, 성별)로 자연스러운 variation 확보.
OpenCharacter/DeepPersona보다 훨씬 단순하지만, 응답 다양성은 보장.
"""
from __future__ import annotations
import random


def generate_persona(country: str, n: int, seed: int = 42, **kwargs) -> list[str]:
    """
    n개의 시스템 프롬프트 생성.
    Cultural prompting — 국적 + 나이/성별만 부여하여 최소한의 variation.
    """
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


# ------------------------------------------------------------------
# 테스트
# ------------------------------------------------------------------
if __name__ == "__main__":
    countries = ["Argentina", "Australia", "Germany", "India", "Kenya", "United States"]
    for c in countries:
        personas = generate_persona(c, 5)
        print(f"\n{'='*60}")
        print(f"Country: {c}")
        for i, p in enumerate(personas):
            print(f"  [{i}] {p[:80]}...")
        unique = len(set(personas))
        print(f"  Total: {len(personas)}, Unique: {unique}")
