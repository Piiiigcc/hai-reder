#!/usr/bin/env python
"""Profile cold-load and generation latency for one debate inference sample.

Run this on the GPU server to separate fixed startup/model-load costs from
actual proposer/critic/judge generation latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/formal_qwen_qlora.yaml")
    parser.add_argument("--input", default="data/processed/splits/smoke_formal_online_window_W5.jsonl")
    parser.add_argument("--proposer-adapter", default="models/adapters/smoke_formal/proposer")
    parser.add_argument("--critic-adapter", default="models/adapters/smoke_formal/critic")
    parser.add_argument("--judge-adapter", default="models/adapters/smoke_formal/judge")
    parser.add_argument("--proposer-prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--critic-prompt", default="prompts/agents/critic_prompt.md")
    parser.add_argument("--judge-prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--pool-path", default=None)
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/metrics/debate_inference_latency_profile.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/arguments without loading models")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            snapshot["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
            snapshot["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 2)
            snapshot["max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            snapshot["max_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    except Exception as exc:
        snapshot["error"] = repr(exc)
    return snapshot


def _measure(name: str, fn: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    _sync_cuda()
    before = _memory_snapshot()
    started = time.perf_counter()
    result = fn()
    _sync_cuda()
    elapsed = time.perf_counter() - started
    return result, {
        "name": name,
        "elapsed_seconds": round(elapsed, 3),
        "memory_before": before,
        "memory_after": _memory_snapshot(),
    }


def _count_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return -1


def main() -> int:
    from haireder.agents.critic import parse_critic_output, render_critic_prompt
    from haireder.agents.judge import compact_transcript_for_judge, parse_judge_output, render_judge_prompt
    from haireder.agents.proposer import parse_proposer_output, render_proposer_prompt
    from haireder.training import load_prompt_template
    from haireder.training.lora_trainer import (
        generate_role_output,
        load_multi_role_bundle,
        require_valid_parsed_output,
        select_role_bundle,
    )
    from haireder.utils.config import load_yaml
    from haireder.utils.io import read_jsonl, write_json

    args = parse_args()
    output_path = _resolve(args.output)
    rows = read_jsonl(_resolve(args.input))
    if not rows:
        raise SystemExit(f"input has no rows: {args.input}")
    if args.row_index < 0 or args.row_index >= len(rows):
        raise SystemExit(f"--row-index must be in [0, {len(rows) - 1}]")
    row = rows[args.row_index]

    report: dict[str, Any] = {
        "started_at": _utc_now(),
        "dry_run": args.dry_run,
        "input": str(_resolve(args.input)),
        "row_index": args.row_index,
        "pair_id": row.get("pair_id"),
        "max_new_tokens": args.max_new_tokens,
        "steps": [],
    }

    if args.dry_run:
        report["status"] = "dry_run"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, report)
        print(f"Profile written: {output_path.relative_to(PROJECT_ROOT)}")
        return 0

    experiment_config = load_yaml(_resolve(args.experiment_config))
    model_config = load_yaml(_resolve(args.model_config))
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]

    proposer_prompt = load_prompt_template(_resolve(args.proposer_prompt))
    critic_prompt = load_prompt_template(_resolve(args.critic_prompt))
    judge_prompt = load_prompt_template(_resolve(args.judge_prompt))

    pool: list[dict[str, Any]] = []
    sbert_model = None
    if args.pool_path:
        pool = read_jsonl(_resolve(args.pool_path))
        from sentence_transformers import SentenceTransformer

        sbert_config = experiment_config.get("confidence", {}).get("sbert", {})
        sbert_model_name = sbert_config.get("model_name", "models/calibration/all-MiniLM-L6-v2")
        sbert_model, step = _measure("load_sbert", lambda: SentenceTransformer(str(_resolve(sbert_model_name))))
        report["steps"].append(step)

    adapter_dirs = {
        "proposer": _resolve(args.proposer_adapter),
        "critic": _resolve(args.critic_adapter),
        "judge": _resolve(args.judge_adapter),
    }
    shared_bundle, step = _measure("load_multi_role_bundle", lambda: load_multi_role_bundle(adapter_dirs, model_config=model_config))
    report["steps"].append(step)
    tokenizer = shared_bundle["tokenizer"]

    proposer_bundle = select_role_bundle(shared_bundle, "proposer")
    critic_bundle = select_role_bundle(shared_bundle, "critic")
    judge_bundle = select_role_bundle(shared_bundle, "judge")

    retrieval_examples: dict[str, str] = {"proposer": "", "critic": "", "judge": ""}
    if pool:
        from haireder.online_learning.retrieval import format_examples_for_role, retrieve_similar_examples

        for role in ("proposer", "critic", "judge"):
            examples, step = _measure(
                f"retrieve_{role}",
                lambda role=role: retrieve_similar_examples(row, pool, role, k=args.retrieval_k, sbert_model=sbert_model),
            )
            report["steps"].append(step)
            retrieval_examples[role] = format_examples_for_role(examples, role)

    proposer_prompt_text = render_proposer_prompt(row, proposer_prompt, retrieval_examples=retrieval_examples["proposer"])
    proposer_raw, step = _measure(
        "generate_proposer",
        lambda: generate_role_output(proposer_bundle, proposer_prompt_text, max_new_tokens=args.max_new_tokens),
    )
    step["prompt_tokens"] = _count_tokens(tokenizer, proposer_prompt_text)
    step["output_tokens"] = _count_tokens(tokenizer, proposer_raw)
    report["steps"].append(step)
    proposer_parsed = require_valid_parsed_output(str(row["pair_id"]), parse_proposer_output(proposer_raw, confidence_map))["parsed_output"]

    critic_prompt_text = render_critic_prompt(
        row,
        json.dumps(proposer_parsed, ensure_ascii=False),
        critic_prompt,
        retrieval_examples=retrieval_examples["critic"],
    )
    critic_raw, step = _measure(
        "generate_critic",
        lambda: generate_role_output(critic_bundle, critic_prompt_text, max_new_tokens=args.max_new_tokens),
    )
    step["prompt_tokens"] = _count_tokens(tokenizer, critic_prompt_text)
    step["output_tokens"] = _count_tokens(tokenizer, critic_raw)
    report["steps"].append(step)
    critic_parsed = require_valid_parsed_output(str(row["pair_id"]), parse_critic_output(critic_raw, confidence_map))["parsed_output"]

    transcript = {
        "pair_id": row["pair_id"],
        "repo": row["repo"],
        "window": row["window"],
        "split": row["split"],
        "rounds": [
            {
                "round_index": 0,
                "proposer": {"raw_output": proposer_raw, "parsed_output": proposer_parsed},
                "critic": {"raw_output": critic_raw, "parsed_output": critic_parsed},
            }
        ],
        "termination_reason": "profile_single_round",
    }
    judge_prompt_text = render_judge_prompt(
        row,
        compact_transcript_for_judge(transcript),
        judge_prompt,
        retrieval_examples=retrieval_examples["judge"],
    )
    judge_raw, step = _measure(
        "generate_judge",
        lambda: generate_role_output(judge_bundle, judge_prompt_text, max_new_tokens=args.max_new_tokens),
    )
    step["prompt_tokens"] = _count_tokens(tokenizer, judge_prompt_text)
    step["output_tokens"] = _count_tokens(tokenizer, judge_raw)
    report["steps"].append(step)
    judge_parsed = parse_judge_output(judge_raw, confidence_map)

    report["status"] = "ok"
    report["prediction"] = {
        "proposer_error": parse_proposer_output(proposer_raw, confidence_map).get("error"),
        "critic_error": parse_critic_output(critic_raw, confidence_map).get("error"),
        "judge_error": judge_parsed.get("error"),
        "judge_output": judge_parsed.get("parsed_output"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    print(f"Profile written: {output_path.relative_to(PROJECT_ROOT)}")
    for step in report["steps"]:
        print(f"{step['name']}: {step['elapsed_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
