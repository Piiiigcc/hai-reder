"""LLM-based consistency validation for augmented requirement pairs."""

from __future__ import annotations

import json
from typing import Any

from haireder.utils.llm_client import APILLMClient, LLMClient

__all__ = ["LLMConsistencyValidator"]


def _issue_text(row: dict[str, Any], side: str) -> str:
    issue = row.get(side) or {}
    return str(issue.get("text") or "").strip()


def _pair_json(row: dict[str, Any]) -> dict[str, str]:
    return {
        "issue_i_text": _issue_text(row, "issue_i"),
        "issue_j_text": _issue_text(row, "issue_j"),
    }


class LLMConsistencyValidator:
    """LLM-based consistency validator for augmented pairs.

    Uses an LLM to assess whether the generated pair preserves the target
    dependency type.
    """

    def __init__(
        self,
        llm_client: LLMClient | APILLMClient,
        prompt_template: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        self.llm_client = llm_client
        self.prompt_template = str(prompt_template)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

    def validate(
        self,
        seed_row: dict[str, Any],
        augmented_row: dict[str, Any],
        target_label: str,
    ) -> dict[str, Any]:
        """Validate whether the augmented pair preserves the target label.

        Returns:
        {
            "is_valid": bool,
            "label_matches": bool,
            "rationale": str,
        }
        """
        prompt = self._build_prompt(seed_row, augmented_row, target_label)
        raw_output = self.llm_client.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return self._parse_llm_output(raw_output)

    def _build_prompt(
        self,
        seed_row: dict[str, Any],
        augmented_row: dict[str, Any],
        target_label: str,
    ) -> str:
        """Fill the validation prompt template."""
        seed_pair_json = json.dumps(_pair_json(seed_row), ensure_ascii=False, indent=2)
        generated_pair_json = json.dumps(_pair_json(augmented_row), ensure_ascii=False, indent=2)
        prompt = self.prompt_template.replace("{{target_label}}", str(target_label))
        prompt = prompt.replace("{{seed_pair_json}}", seed_pair_json)
        prompt = prompt.replace("{{generated_pair_json}}", generated_pair_json)
        return prompt

    @staticmethod
    def _candidate_json_strings(raw_output: str) -> list[str]:
        text = str(raw_output or "").strip()
        if not text:
            return []

        candidates = [text]

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                if lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                fenced_text = "\n".join(lines).strip()
                if fenced_text and fenced_text not in candidates:
                    candidates.append(fenced_text)

        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            brace_text = text[start : end + 1].strip()
            if brace_text and brace_text not in candidates:
                candidates.append(brace_text)

        return candidates

    def _parse_llm_output(self, raw_output: str) -> dict[str, Any]:
        """Parse the LLM output JSON."""
        fallback = {"is_valid": False, "label_matches": False, "rationale": "parse_error"}

        for candidate in self._candidate_json_strings(raw_output):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            return {
                "is_valid": parsed.get("is_valid") is True,
                "label_matches": parsed.get("label_matches") is True,
                "rationale": str(parsed.get("rationale", "")).strip(),
            }

        return fallback
