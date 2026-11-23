from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "debate" / "run_debate_inference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_debate_inference", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_valid_turn_retries_then_returns_tagged_fallback() -> None:
    module = _load_module()
    budgets: list[int] = []

    def generate_output(_bundle, _prompt_text, *, max_new_tokens: int) -> str:
        budgets.append(max_new_tokens)
        return "{\"label\": \"unterminated"

    def parse_output(_raw: str, _confidence_map: dict[str, float]) -> dict:
        return {"parsed_output": None, "error": "JSONDecodeError('Unterminated string')"}

    def validate_output(pair_id: str, parsed: dict) -> dict:
        raise ValueError(f"Failed to parse model output for {pair_id}: {parsed['error']}")

    result = module._generate_valid_turn(
        pair_id="p1",
        bundle=object(),
        prompt_text="prompt",
        max_new_tokens=64,
        confidence_map={"very_low": 0.1},
        parse_fn=parse_output,
        generate_output_fn=generate_output,
        validate_fn=validate_output,
        fallback_label_key="label",
        diagnostic_fallback=True,
    )

    assert budgets == [64, 128]
    assert result["parsed_output"]["is_fallback_generation"] is True
    assert result["parsed_output"]["fallback_reason"] == "all_parse_attempts_exhausted"
    assert result["parsed_output"]["verbal_confidence"] == "very_low"
