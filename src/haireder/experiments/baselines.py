"""Supervision-matched baselines for the held-out pair cohort."""

import json
import math
from datetime import datetime
from pathlib import Path

from haireder.baselines.tfidf_logreg import TfidfLogRegBaseline
from haireder.taxonomy import LABELS
from haireder.utils.io import write_json, write_jsonl
from haireder.utils.model_versions import revision_kwargs
from .evaluation import reject_substituted_turns, validate_predictions, evaluation
from .review import fingerprint


def cumulative_supervision(initial, accepted, test):
    if any(r["split"] != "train" for r in initial) or any(
        r["split"] != "test" for r in test
    ):
        raise ValueError("Invalid training or held-out split")
    rows = list(initial)
    for r in accepted:
        reject_substituted_turns(r)
        if (
            r["split"] != "online"
            or r.get("analyst_simulation")
            or r.get("decision_path")
            not in ("low_review", "medium_review", "medium_promoted", "high_auto")
            or r.get("accepted_label") not in LABELS
        ):
            raise ValueError("Unverified online supervision")
        if r["decision_path"] in ("low_review", "medium_review"):
            receipt = r.get("review_detail") or {}
            if (
                r.get("human_reviewed") is not True
                or receipt.get("pair_id") != r["pair_id"]
                or receipt.get("accepted_label") != r["accepted_label"]
                or not all(
                    str(receipt.get(k) or "").strip()
                    for k in (
                        "reviewer",
                        "reviewed_at",
                        "rationale",
                        "item_hash",
                        "batch_id",
                    )
                )
                or datetime.fromisoformat(
                    receipt["reviewed_at"].replace("Z", "+00:00")
                ).tzinfo
                is None
            ):
                raise ValueError("Human supervision requires a signed matching review")
        elif (
            r.get("human_reviewed") is not False
            or r.get("predicted_label") != r["accepted_label"]
        ):
            raise ValueError(
                "Automatic supervision must preserve the actual model prediction"
            )
        rows.append(dict(r, label=r["accepted_label"]))
    keys = {r["pair_id"] for r in rows}
    unordered = lambda r: (r["repo"], *sorted((r["issue_i"]["id"], r["issue_j"]["id"])))
    if (
        len(keys) != len(rows)
        or keys & {r["pair_id"] for r in test}
        or {unordered(r) for r in rows} & {unordered(r) for r in test}
    ):
        raise ValueError("Duplicate supervision or held-out overlap")
    return rows


def prediction(row, probabilities, name):
    if (
        len(probabilities) != len(LABELS)
        or any(not 0 <= p <= 1 for p in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
    ):
        raise ValueError(f"Invalid class probabilities for {row['pair_id']}")
    idx = max(range(len(LABELS)), key=probabilities.__getitem__)
    return dict(
        pair_id=row["pair_id"],
        repo=row["repo"],
        window=row["window"],
        split=row["split"],
        issue_i=row["issue_i"],
        issue_j=row["issue_j"],
        gold_label=row["label"],
        predicted_label=LABELS[idx],
        confidence=float(probabilities[idx]),
        probabilities=dict(zip(LABELS, map(float, probabilities))),
        method=name,
    )


def bert_predictions(
    training,
    test,
    model_name,
    *,
    epochs=3,
    batch_size=8,
    device="cpu",
    max_length=256,
    seed=42,
):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs(model_name))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **revision_kwargs(model_name),
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def inputs(rows):
        return tokenizer(
            [r["issue_i"]["text"] for r in rows],
            [r["issue_j"]["text"] for r in rows],
            return_tensors="pt",
            padding=True,
            truncation="longest_first",
            max_length=max_length,
        ).to(device)

    model.train()
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(training), generator=generator).tolist()
        for i in range(0, len(order), batch_size):
            batch = [training[j] for j in order[i : i + batch_size]]
            labels = torch.tensor(
                [LABELS.index(r["label"]) for r in batch], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs(batch), labels=labels).loss
            loss.backward()
            optimizer.step()
            history.append(loss.item())
    model.eval()
    results = []
    with torch.inference_mode():
        for i in range(0, len(test), batch_size):
            batch = test[i : i + batch_size]
            probs = torch.softmax(model(**inputs(batch)).logits, dim=1).cpu().tolist()
            results.extend(
                prediction(r, p, "bert_finetuned")
                for r, p in zip(batch, probs, strict=True)
            )
    return results, dict(
        model=model_name,
        **revision_kwargs(model_name),
        epochs=epochs,
        steps=len(history),
        initial_loss=history[0],
        final_loss=history[-1],
        seed=seed,
    )


def zero_prefix(tokenizer, row):
    texts = [
        tokenizer.decode(
            tokenizer.encode(row[k]["text"], add_special_tokens=False)[:750]
        )
        for k in ("issue_i", "issue_j")
    ]
    question = (
        "Select the dependency label. Reply with its number only. "
        + json.dumps(dict(enumerate(LABELS, 1)))
        + "\nRequirement A: "
        + texts[0]
        + "\nRequirement B: "
        + texts[1]
    )
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(chat, add_special_tokens=False)


def zero_predictions(test, model_name, *, device="cpu", model_config=None):
    """Compare summed conditional log probabilities of the six numeric choice tokens."""
    from haireder.training.lora_trainer import _build_inference_model_kwargs
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs(model_name))
    kwargs = _build_inference_model_kwargs(model_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **revision_kwargs(model_name), **kwargs
    )
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    result = []
    answers = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 7)]
    for row in test:
        prefix = zero_prefix(tokenizer, row)
        with torch.inference_mode():
            if all(len(ids) == 1 for ids in answers):
                logits = model(torch.tensor([prefix], device=model.device)).logits[
                    0, -1
                ]
                scores = logits[[ids[0] for ids in answers]]
            else:
                scores = []
                for ids in answers:
                    sequence = torch.tensor([prefix + ids], device=model.device)
                    logp = (
                        model(sequence).logits[0, len(prefix) - 1 : -1].log_softmax(-1)
                    )
                    scores.append(
                        logp[torch.arange(len(ids), device=model.device), ids].sum()
                    )
                scores = torch.stack(scores)
            result.append(
                prediction(
                    row,
                    scores.float().softmax(0).cpu().tolist(),
                    "zero_shot_choice_likelihood",
                )
            )
    return result, dict(
        model=model_name,
        **revision_kwargs(model_name),
        adapter=None,
        train_examples=0,
        choices=dict(enumerate(LABELS, 1)),
    )


def run_baseline(
    method,
    initial,
    accepted,
    test,
    directory,
    *,
    device="cpu",
    model_cfg=None,
    bert_model="google-bert/bert-base-uncased",
    zero_model="Qwen/Qwen2.5-7B-Instruct",
    epochs=3,
    batch_size=8,
    test_total=None,
):
    test_total = len(test) if test_total is None else test_total
    if not 0 < len(test) <= test_total:
        raise ValueError("Invalid baseline evaluation coverage")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    training = (
        [] if method == "zero_shot" else cumulative_supervision(initial, accepted, test)
    )
    if method == "tfidf":
        model = TfidfLogRegBaseline().fit(training)
        rows = model.predict_rows(test)
        detail = {
            "features_per_side": len(model.pipeline.named_steps["tfidf"].vocabulary_)
        }
    elif method == "bert":
        rows, detail = bert_predictions(
            training,
            test,
            bert_model,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )
    elif method == "zero_shot":
        rows, detail = zero_predictions(
            test, zero_model, device=device, model_config=model_cfg
        )
    else:
        raise ValueError(method)
    validate_predictions(test, rows)
    write_jsonl(directory / "predictions.jsonl", rows)
    write_json(directory / "metrics.json", evaluation(rows))
    write_json(
        directory / "supervision.json",
        dict(
            method=method,
            training_count=len(training),
            train_hash=fingerprint([(r["pair_id"], r["label"]) for r in training]),
            test_hash=fingerprint(test),
            test_total=test_total,
            evaluated=len(rows),
            test_scope="full" if len(rows) == test_total else "diagnostic_subset",
            **detail,
        ),
    )


def zero_mlx_predictions(test, model_name, *, batch_size=4):
    """Score all six choices on the quantized Qwen backbone using causal right padding."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from haireder.utils.llm_client import MLXClient

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    mx.set_cache_limit(96 * 2**20)
    mx.set_memory_limit(
        min(7 * 2**30, mx.device_info()["max_recommended_working_set_size"])
    )
    model, tokenizer = MLXClient(model_name).bundle()
    if model.model_type != "qwen2":
        raise ValueError("The Apple acceleration path targets the Qwen2 family")
    tokens = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 7)]
    if any(len(t) != 1 for t in tokens):
        raise ValueError("Choice labels must be single tokens for batched MLX scoring")
    words = [t[0] for t in tokens]
    sequences = [zero_prefix(tokenizer, r) for r in test]
    order = sorted(range(len(test)), key=lambda i: len(sequences[i]))
    out = [None] * len(order)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        lengths = [len(sequences[i]) for i in indices]
        width = max(lengths)
        ids = mx.array(
            [
                sequences[i] + [tokenizer.eos_token_id] * (width - len(sequences[i]))
                for i in indices
            ]
        )
        cache = make_prompt_cache(model)
        # Keep long Jira prompts bounded while preserving all causal positions.
        for begin in range(0, width, 256):
            end = min(begin + 256, width)
            hidden = model.model(ids[:, begin:end], cache=cache)
            finished = [j for j, length in enumerate(lengths) if begin < length <= end]
            if finished:
                last = hidden[
                    mx.array(finished),
                    mx.array([lengths[j] - begin - 1 for j in finished]),
                ]
                logits = (
                    model.model.embed_tokens.as_linear(last)
                    if model.args.tie_word_embeddings
                    else model.lm_head(last)
                )
                probabilities = mx.softmax(
                    logits[:, words].astype(mx.float32), axis=-1
                ).tolist()
                for j, prob in zip(finished, probabilities, strict=True):
                    i = indices[j]
                    out[i] = prediction(
                        test[i], prob, "zero_shot_choice_likelihood_mlx4bit"
                    )
            mx.eval([layer.state for layer in cache])
        del cache, hidden
    return out, dict(
        model=model_name,
        **revision_kwargs(model_name),
        device="Apple MLX",
        quantization="4-bit group quantized",
        adapter=None,
        train_examples=0,
        choices=dict(enumerate(LABELS, 1)),
        causal_prefill_block=256,
    )
