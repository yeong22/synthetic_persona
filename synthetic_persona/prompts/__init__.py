"""
Persona prompting 모듈.
3가지 전략 모두 동일한 인터페이스:
  generate_persona(country, n, ...) → list[str]   # n개 시스템 프롬프트

공유 유틸리티 ��수.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def format_question_prompt(item: dict) -> str:
    """질문 dict를 사용자 프롬프트 문자열로 변환.

    DeepPersona 스타일: item에 'prompt_suffix'가 있으면 그대로 사용 (WVS).
    Big Five: "Rate how accurately this describes you" + 1-5 스케일.
    Privacy/기타: 기존 라벨 기반 포맷.
    """
    scale_min, scale_max = item["scale"]

    # DeepPersona WVS 스타일: prompt_suffix 사용
    if "prompt_suffix" in item:
        return (
            f"{item['text']}\n\n"
            f"{item['prompt_suffix']}\n\n"
            f"Answer with ONLY a single integer from {scale_min} to {scale_max}."
        )

    # Big Five 스타일
    if item.get("factor"):
        return (
            f"Rate how accurately the following statement describes you.\n\n"
            f"\"{item['text']}\"\n\n"
            f"Scale: 1=Very Inaccurate, 2=Moderately Inaccurate, "
            f"3=Neither Accurate Nor Inaccurate, 4=Moderately Accurate, "
            f"5=Very Accurate\n\n"
            f"Answer with ONLY a single integer from 1 to 5."
        )

    # 기본: 라벨 기반 (Privacy 등)
    labels_str = ", ".join(f"{k}={v}" for k, v in sorted(item["labels"].items()))
    return (
        f"{item['text']}\n\n"
        f"Scale: {labels_str}\n\n"
        f"Answer with ONLY a single integer from {scale_min} to {scale_max}."
    )
