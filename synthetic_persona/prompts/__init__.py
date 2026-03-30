"""
Persona prompting 모듈.
3가지 전략 모두 동일한 인터페이스:
  generate_persona(country, n, ...) → list[str]   # n개 시스템 프롬프트
  run_survey(persona, questions, client) → dict    # 설문 응답

공유 유틸리티 함수.
"""
import sys
from pathlib import Path

# engine 임포트를 위한 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def format_question_prompt(item: dict) -> str:
    """WVS/Privacy 질문 dict를 사용자 프롬프트 문자열로 변환."""
    scale_min, scale_max = item["scale"]
    label_min = item["labels"].get(scale_min, "")
    label_max = item["labels"].get(scale_max, "")

    # 모든 라벨이 있으면 전부 표시, 아니면 양 끝만
    if len(item["labels"]) == (scale_max - scale_min + 1):
        labels_str = ", ".join(f"{k}={v}" for k, v in sorted(item["labels"].items()))
        return (
            f"{item['text']}\n\n"
            f"Scale: {labels_str}\n\n"
            f"Answer with ONLY a single integer from {scale_min} to {scale_max}."
        )
    else:
        return (
            f"{item['text']}\n\n"
            f"Scale: {scale_min}={label_min} ... {scale_max}={label_max}\n\n"
            f"Answer with ONLY a single integer from {scale_min} to {scale_max}."
        )


def run_survey(persona: str, questions: dict, client) -> dict:
    """
    하나의 persona 시스템 프롬프트로 설문 전체를 수행.

    Args:
        persona:   시스템 프롬프트 문자열
        questions: {item_key: item_dict} (config의 WVS_ITEMS 또는 PRIVACY_ITEMS)
        client:    engine.llm_client.LocalLLM 인스턴스

    Returns:
        {item_key: int|None} 응답 딕셔너리
    """
    responses = {}
    for item_key, item in questions.items():
        user_prompt = format_question_prompt(item)
        scale_min, scale_max = item["scale"]
        value = client.query_int(
            system_prompt=persona,
            user_prompt=user_prompt,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        responses[item_key] = value
    return responses
