"""LLM-based pair-aware augmentation generator."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Any

from haireder.taxonomy import Taxonomy
from haireder.utils.llm_client import APILLMClient, LLMClient

__all__ = ["LLMPairAugmenter"]


def _issue_text(row: dict[str, Any], side: str) -> str:
    issue = row.get(side) or {}
    return str(issue.get("text") or "").strip()


def _first_json_object(text: str) -> str:
    """Extract first JSON object from text."""
    start = text.find("{")
    if start == -1:
        return text.strip()
    candidate = text[start:]
    try:
        _, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return candidate.strip()
    return candidate[:end]


class LLMPairAugmenter:
    """LLM-based pair-aware augmentation generator."""

    def __init__(
        self,
        llm_client: LLMClient | APILLMClient,
        prompt_template: str,
        taxonomy: Taxonomy,
        reference_pairs_per_prompt: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.llm_client = llm_client
        self.prompt_template = str(prompt_template)
        self.taxonomy = taxonomy
        self.reference_pairs_per_prompt = int(reference_pairs_per_prompt)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

    def generate(
        self,
        seed_row: dict[str, Any],
        target_label: str,
        variant_index: int,
        reference_pool: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Generate one augmented row from a seed pair. Returns None on failure."""
        prompt = self._build_prompt(seed_row, target_label, variant_index, reference_pool or [])
        try:
            raw_output = self.llm_client.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception:
            return None
        parsed = self._parse_llm_output(raw_output)
        if parsed is None:
            return None
        response_label = str(parsed.get("target_label", "")).strip()
        if response_label and response_label != target_label:
            return None
        return self._build_augmented_row(seed_row, parsed, target_label, variant_index)

    def _build_prompt(
        self,
        seed_row: dict[str, Any],
        target_label: str,
        variant_index: int,
        reference_pool: list[dict[str, Any]],
    ) -> str:
        """Fill the augmentation prompt template."""
        label_definition = self.taxonomy.label_definitions.get(target_label, "")
        issue_i_text = _issue_text(seed_row, "issue_i")
        issue_j_text = _issue_text(seed_row, "issue_j")

        same_class = [
            r
            for r in reference_pool
            if r.get("label") == target_label and r.get("pair_id") != seed_row.get("pair_id")
        ]
        if same_class:
            refs = random.Random(f"{seed_row['pair_id']}:{variant_index}").sample(same_class, min(self.reference_pairs_per_prompt, len(same_class)))
        else:
            refs = []
        ref_texts = []
        for i, ref in enumerate(refs, 1):
            ref_texts.append(
                f"Reference pair {i}:\nRequirement A: {_issue_text(ref, 'issue_i')}\nRequirement B: {_issue_text(ref, 'issue_j')}"
            )
        ref_block = "\n\n".join(ref_texts)

        prompt = self.prompt_template.replace("{{target_label}}", str(target_label))
        prompt = prompt.replace("{{target_label_definition}}", str(label_definition))
        prompt = prompt.replace("{{issue_i_text}}", issue_i_text)
        prompt = prompt.replace("{{issue_j_text}}", issue_j_text)
        if "{{reference_pairs}}" in prompt:
            prompt = prompt.replace("{{reference_pairs}}", ref_block)
        elif ref_block:
            prompt = prompt.rstrip() + "\n\n" + ref_block
        return prompt

    def _parse_llm_output(self, raw_output: str) -> dict[str, Any] | None:
        """Parse LLM JSON output."""
        candidate = _first_json_object(raw_output.strip())
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        if "augmented_issue_i_text" not in parsed or "augmented_issue_j_text" not in parsed:
            return None
        if not str(parsed["augmented_issue_i_text"]).strip() or not str(parsed["augmented_issue_j_text"]).strip():
            return None
        return parsed

    def _build_augmented_row(
        self,
        seed_row: dict[str, Any],
        parsed: dict[str, Any],
        target_label: str,
        variant_index: int,
    ) -> dict[str, Any]:
        """Build schema-compatible augmented row (same format as build_rule_based_augmented_pair)."""
        augmented = deepcopy(seed_row)
        for field in ("transcript","review_detail","accepted_label","gold_label","review_source","analyst_status",
                      "human_reviewed","analyst_simulation","confidence","confidence_region","decision_path",
                      "predicted_label","rationale","s1","s2","s3","s3_neighbors","calibration_mode"):
            augmented.pop(field,None)
        seed_pair_id = str(seed_row["pair_id"])
        issue_i_id = str((seed_row.get("issue_i") or {}).get("id", "i"))
        issue_j_id = str((seed_row.get("issue_j") or {}).get("id", "j"))

        augmented["pair_id"] = f"AUG-{seed_pair_id}-{variant_index:03d}"
        augmented["label"] = target_label
        augmented["source"] = "pair_aware_llm_augmentation"
        augmented["is_augmented"] = True
        augmented["source_seed_pair_id"] = seed_pair_id
        augmented["augmentation"] = {
            "generator": "llm_v1",
            "generator_model": getattr(self.llm_client, "model_name", type(self.llm_client).__name__),
            "variant_index": variant_index,
            "target_label": target_label,
            "source_seed_pair_id": seed_pair_id,
            "pair_sim_backend": "llm_generated",
            "preservation_note": str(parsed.get("preservation_note", "")),
        }
        augmented["issue_i"] = dict(seed_row.get("issue_i") or {})
        augmented["issue_i"]["id"] = f"{issue_i_id}-AUG{variant_index:03d}"
        augmented["issue_i"]["text"] = str(parsed["augmented_issue_i_text"]).strip()
        augmented["issue_j"] = dict(seed_row.get("issue_j") or {})
        augmented["issue_j"]["id"] = f"{issue_j_id}-AUG{variant_index:03d}"
        augmented["issue_j"]["text"] = str(parsed["augmented_issue_j_text"]).strip()
        return augmented
