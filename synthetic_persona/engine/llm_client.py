"""
vLLM offline inference client.
모델을 프로세스 내에서 직접 로드하여 추론. 서버 불필요.
프로세스 종료 시 GPU 메모리 자동 해제.
"""
import re
import json
import logging

from vllm import LLM, SamplingParams

from config.experiment_config import (
    MODEL_ID, TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN, MAX_TOKENS, TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LocalLLM:
    """vLLM offline inference — no server, GPU freed on process exit."""

    def __init__(
        self,
        model: str = MODEL_ID,
        tensor_parallel_size: int = TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        max_model_len: int = MAX_MODEL_LEN,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        logger.info("Loading model %s (TP=%d)...", model, tensor_parallel_size)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="auto",
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------
    def query(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """단일 chat completion. raw text 반환."""
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = self.llm.chat([conversation], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def query_int(
        self,
        system_prompt: str,
        user_prompt: str,
        scale_min: int,
        scale_max: int,
        **kwargs,
    ) -> int | None:
        """단일 쿼리 후 [scale_min, scale_max] 범위 정수 파싱."""
        raw = self.query(system_prompt, user_prompt, **kwargs)
        return self.parse_int(raw, scale_min, scale_max)

    # ------------------------------------------------------------------
    # Batch query (vLLM 배치 추론 — 핵심 성능 이점)
    # ------------------------------------------------------------------
    def query_batch(self, conversations: list[list[dict]], **kwargs) -> list[str]:
        """
        배치 chat completion.
        conversations: [conversation, ...] 각 conversation = [{"role":..., "content":...}, ...]
        """
        outputs = self.llm.chat(conversations, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def query_int_batch(
        self,
        conversations: list[list[dict]],
        scale_min: int,
        scale_max: int,
    ) -> list[int | None]:
        """배치 쿼리 후 정수 파싱."""
        raw_responses = self.query_batch(conversations)
        return [self.parse_int(r, scale_min, scale_max) for r in raw_responses]

    # ------------------------------------------------------------------
    # Parsing (static — 모델 없이도 사용 가능)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_int(text: str, scale_min: int, scale_max: int) -> int | None:
        """텍스트에서 [scale_min, scale_max] 범위의 첫 번째 정수 추출."""
        if not text:
            return None
        numbers = re.findall(r"\d+", text)
        for n_str in numbers:
            n = int(n_str)
            if scale_min <= n <= scale_max:
                return n
        return None

    @staticmethod
    def parse_json(text: str) -> dict | list | None:
        """텍스트에서 JSON 객체/배열 추출. markdown fence 포함 대응."""
        if not text:
            return None
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        if hasattr(self, "llm"):
            del self.llm


# ======================================================================
# Test / 검증
# ======================================================================
def _test_parse():
    """parse_int / parse_json 단위 테스트 (모델 불필요)."""
    print("=" * 60)
    print("parse_int / parse_json 단위 테스트")
    print("=" * 60)

    all_pass = True

    # parse_int
    print("\n[1] parse_int 테스트...")
    int_cases = [
        ("5", 1, 10, 5),
        ("I would say 7.", 1, 10, 7),
        ("Rating: 3/10", 1, 10, 3),
        ("no number here", 1, 10, None),
        ("99", 1, 10, None),
        ("The answer is 42", 1, 7, None),
    ]
    for text, lo, hi, expected in int_cases:
        result = LocalLLM.parse_int(text, lo, hi)
        status = "OK" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: parse_int('{text}', {lo}, {hi}) = {result} (expected {expected})")

    # parse_json
    print("\n[2] parse_json 테스트...")
    json_cases = [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"b": 2}\n```', {"b": 2}),
        ('Here is the result: {"c": 3} end', {"c": 3}),
        ("no json here", None),
        ('[1, 2, 3]', [1, 2, 3]),
    ]
    for text, expected in json_cases:
        result = LocalLLM.parse_json(text)
        status = "OK" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: parse_json('{text[:40]}') = {result}")

    print("\n" + "=" * 60)
    print(f"파싱 테스트: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return all_pass


def _test_inference():
    """모델 로드 + 추론 테스트 (GPU 필요)."""
    print("\n" + "=" * 60)
    print("LocalLLM 추론 테스트 (GPU 필요)")
    print("=" * 60)

    llm = LocalLLM()

    # 단일 쿼리
    print("\n[3] 단일 쿼리...")
    resp = llm.query(
        system_prompt="You are a helpful assistant. Answer briefly.",
        user_prompt="What is 2 + 3? Answer with just the number.",
    )
    print(f"  응답: '{resp}'")

    # 정수 파싱 쿼리
    print("\n[4] 정수 파싱 쿼리 (Likert 1-7)...")
    val = llm.query_int(
        system_prompt="You are a survey respondent. Answer with a single integer.",
        user_prompt="How satisfied are you with your life? (1=Very dissatisfied, 7=Very satisfied)",
        scale_min=1,
        scale_max=7,
    )
    print(f"  파싱된 값: {val}")

    # 배치 쿼리
    print("\n[5] 배치 쿼리 (3건)...")
    conversations = [
        [{"role": "system", "content": "Answer with just a number."},
         {"role": "user", "content": "What is 1+1?"}],
        [{"role": "system", "content": "Answer with just a number."},
         {"role": "user", "content": "What is 2+2?"}],
        [{"role": "system", "content": "Answer with just a number."},
         {"role": "user", "content": "What is 3+3?"}],
    ]
    results = llm.query_batch(conversations)
    for i, r in enumerate(results):
        print(f"  [{i}] '{r}'")

    print("\n추론 테스트 완료.")
    del llm


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _test_parse()

    if "--no-gpu" not in sys.argv:
        _test_inference()
    else:
        print("\n(--no-gpu: 추론 테스트 건너뜀)")
