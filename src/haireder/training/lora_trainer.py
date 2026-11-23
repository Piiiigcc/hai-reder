from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haireder.agents.proposer import parse_proposer_output, render_proposer_prompt
from haireder.utils.env import detect_torch_backend
from haireder.utils.model_versions import revision_kwargs


def _load_transformer_stack():
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, DataCollatorForSeq2Seq

    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "default_data_collator": default_data_collator,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
    }


def _offline_kwargs() -> dict[str, Any]:
    import os
    value = os.environ.get("HF_HUB_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", ""))
    if value.strip().upper() in {"1", "TRUE", "YES", "ON"}:
        return {"local_files_only": True}
    return {}


def choose_base_model_name(model_config: dict[str, Any]) -> str:
    stack = _load_transformer_stack()
    auto_tokenizer = stack["AutoTokenizer"]
    primary = str(model_config["base_model"]["primary"])
    fallbacks = [str(name) for name in model_config["base_model"].get("fallbacks", [])]
    candidates = [primary, *fallbacks]
    errors: list[str] = []
    offline = _offline_kwargs()
    for candidate in candidates:
        try:
            auto_tokenizer.from_pretrained(candidate, **revision_kwargs(candidate), **offline)
            return candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")
    raise RuntimeError("No configured base model could be loaded: " + " | ".join(errors))


def _prepare_device(model_config: dict[str, Any]) -> str:
    available = detect_torch_backend()
    options = model_config.get("device_preference") or [available["selected_backend"]]
    for preferred in options:
        if preferred == "cpu" or available.get(f"{preferred}_available") or preferred == available.get("selected_backend"):
            return preferred
    raise RuntimeError(f"No requested device available: {options}")


def _build_quantization_config(model_config: dict[str, Any]):
    quant = model_config.get("quantization", {})
    if not quant.get("enabled"):
        return None

    from transformers import BitsAndBytesConfig
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(str(quant.get("bnb_4bit_compute_dtype", "bfloat16")), torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=quant.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


def _build_inference_model_kwargs(model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not model_config:
        return {}
    quantization_config = _build_quantization_config(model_config)
    if quantization_config is None:
        return {}
    return {
        "quantization_config": quantization_config,
        "device_map": "auto",
    }


def _training_args_from_config(train_section: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ("fp16", "bf16", "gradient_checkpointing", "warmup_ratio", "lr_scheduler_type"):
        if key in train_section:
            kwargs[key] = train_section[key]
    return kwargs


def _enable_gradient_checkpointing_if_configured(model: Any, train_section: dict[str, Any]) -> None:
    if not bool(train_section.get("gradient_checkpointing")):
        return
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def select_inference_device(metadata_device: str | None = None) -> str:
    available = detect_torch_backend()
    selected_backend = str(available.get("selected_backend", "cpu"))
    if selected_backend and selected_backend != "unavailable":
        return selected_backend
    if metadata_device:
        return str(metadata_device)
    return "cpu"


def require_valid_parsed_output(pair_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    if parsed.get("error") is not None:
        raise ValueError(f"Failed to parse model output for {pair_id}: {parsed['error']}")
    return parsed


def merge_prompt_and_target_ids(
    prompt_ids: list[int],
    target_ids: list[int],
    max_length: int,
    pad_token_id: int,
) -> dict[str, list[int]]:
    if len(target_ids) >= max_length:
        kept_prompt_ids: list[int] = []
        kept_target_ids = target_ids[:max_length]
    else:
        prompt_budget = max_length - len(target_ids)
        kept_prompt_ids = prompt_ids[:prompt_budget]
        kept_target_ids = target_ids
    input_ids = kept_prompt_ids + kept_target_ids
    labels = ([-100] * len(kept_prompt_ids)) + kept_target_ids
    attention_mask = [1] * len(input_ids)
    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids = input_ids + ([pad_token_id] * pad_length)
        labels = labels + ([-100] * pad_length)
        attention_mask = attention_mask + ([0] * pad_length)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _tokenize_examples(examples: list[dict[str, Any]], tokenizer, max_length: int):
    from .context import fit_prompt
    stack = _load_transformer_stack()
    dataset = stack["Dataset"].from_list(examples)

    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        target_text = row["target_text"] + (tokenizer.eos_token or "")
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(target_ids) >= max_length:
            raise ValueError(f"Training target exceeds context for {row.get('pair_id')}")
        prompt = fit_prompt(row["prompt_text"], tokenizer, max_length-len(target_ids))
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        length = len(prompt_ids) + len(target_ids)
        return merge_prompt_and_target_ids(prompt_ids,target_ids,max_length=length,pad_token_id=int(tokenizer.pad_token_id))


    return dataset.map(tokenize_row, remove_columns=dataset.column_names)


def _lora_config(model_config: dict[str, Any]):
    stack = _load_transformer_stack()
    lora_section = model_config["lora"]
    kwargs: dict[str, Any] = {
        "r": int(lora_section["r"]),
        "lora_alpha": int(lora_section["alpha"]),
        "lora_dropout": float(lora_section["dropout"]),
        "task_type": stack["TaskType"].CAUSAL_LM,
    }
    target_modules = lora_section.get("target_modules")
    if target_modules == "auto":
        kwargs["target_modules"] = "all-linear"
    elif target_modules:
        kwargs["target_modules"] = target_modules
    return stack["LoraConfig"](**kwargs)


def latest_trainer_checkpoint(adapter_dir: str | Path) -> Path | None:
    """Return the newest Hugging Face Trainer checkpoint under an adapter dir."""
    trainer_state_dir = Path(adapter_dir).expanduser().resolve() / "trainer_state"
    if not trainer_state_dir.exists():
        return None
    candidates = []
    for path in trainer_state_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def resolve_resume_checkpoint(
    adapter_dir: str | Path,
    resume_from_checkpoint: str | Path | None,
) -> Path | None:
    """Resolve resume checkpoint policy.

    `auto` means the newest `trainer_state/checkpoint-*`; `none`/empty disables
    resume; any other value is treated as an explicit checkpoint path.
    """
    if resume_from_checkpoint is None:
        return None
    value = str(resume_from_checkpoint)
    if value == "" or value.lower() in {"none", "false", "0"}:
        return None
    if value.lower() == "auto":
        return latest_trainer_checkpoint(adapter_dir)
    return Path(resume_from_checkpoint).expanduser().resolve()


def train_role_adapter(
    examples: list[dict[str, Any]],
    model_config: dict[str, Any],
    adapter_dir: str | Path,
    max_steps: int | None = None,
    resume_from_checkpoint: str | Path | None = "auto",
    checkpoint_save_steps: int = 25,
    checkpoint_save_total_limit: int = 3,
    parent_adapter: str | Path | None = None,
):
    stack = _load_transformer_stack()
    auto_model = stack["AutoModelForCausalLM"]
    auto_tokenizer = stack["AutoTokenizer"]
    trainer_cls = stack["Trainer"]
    training_args_cls = stack["TrainingArguments"]
    default_data_collator = stack["default_data_collator"]
    get_peft_model = stack["get_peft_model"]

    chosen_model = choose_base_model_name(model_config)
    device = _prepare_device(model_config)
    offline = _offline_kwargs()
    tokenizer = auto_tokenizer.from_pretrained(chosen_model, **revision_kwargs(chosen_model), **offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quantization_config(model_config)
    model_kwargs = {"quantization_config": quantization_config} if quantization_config else {}
    model_kwargs.update(offline)
    model = auto_model.from_pretrained(chosen_model, **revision_kwargs(chosen_model), **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    train_section = model_config["training"]
    _enable_gradient_checkpointing_if_configured(model, train_section)
    if quantization_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(train_section.get("gradient_checkpointing")))
    if parent_adapter is None:
        model = get_peft_model(model, _lora_config(model_config))
    else:
        if load_adapter_metadata(parent_adapter)["chosen_model_name"] != chosen_model:
            raise ValueError("Parent adapter backbone differs from the training backbone")
        model = stack["PeftModel"].from_pretrained(model, parent_adapter, is_trainable=True)

    train_dataset = _tokenize_examples(
        examples,
        tokenizer,
        int(model_config["tokenization"]["max_seq_length"]),
    )

    adapter_path = Path(adapter_dir).expanduser().resolve()
    adapter_path.mkdir(parents=True, exist_ok=True)
    resolved_resume_checkpoint = resolve_resume_checkpoint(adapter_path, resume_from_checkpoint)
    save_steps = max(1, int(checkpoint_save_steps))
    if max_steps is not None and int(max_steps) > 0:
        save_steps = min(save_steps, int(max_steps))
    training_args = training_args_cls(
        output_dir=str(adapter_path / "trainer_state"),
        per_device_train_batch_size=int(train_section["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_section["gradient_accumulation_steps"]),
        learning_rate=float(train_section["learning_rate"]),
        num_train_epochs=float(train_section["epochs"]),
        max_steps=int(max_steps) if max_steps is not None else -1,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=max(1, int(checkpoint_save_total_limit)),
        report_to=[],
        remove_unused_columns=False,
        use_cpu=device == "cpu",
        seed=int(train_section["seed"]),
        **_training_args_from_config(train_section),
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=(stack["DataCollatorForSeq2Seq"](tokenizer,label_pad_token_id=-100)
                       if "DataCollatorForSeq2Seq" in stack else default_data_collator),
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resolved_resume_checkpoint) if resolved_resume_checkpoint else None
    )
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metadata = {
        "chosen_model_name": chosen_model,
        "upstream_revision": revision_kwargs(chosen_model).get("revision"),
        "device": device,
        "train_examples": len(examples),
        "parent_adapter": str(parent_adapter) if parent_adapter else None,
        "max_steps": max_steps,
        "max_seq_length": int(model_config["tokenization"]["max_seq_length"]),
        "resume_from_checkpoint": str(resolved_resume_checkpoint) if resolved_resume_checkpoint else None,
        "checkpoint_save_steps": save_steps,
        "checkpoint_save_total_limit": max(1, int(checkpoint_save_total_limit)),
        "train_metrics": dict(train_result.metrics),
        "optimizer_steps": getattr(train_result, "global_step", None),
    }
    from haireder.data.formal import file_hash
    if (adapter_path/"adapter_model.safetensors").exists():
        metadata["adapter_sha256"]=file_hash(adapter_path/"adapter_model.safetensors")
    if parent_adapter:
        metadata["parent_weights_sha256"]=file_hash(Path(parent_adapter)/"adapter_model.safetensors")
    (adapter_path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def load_adapter_metadata(adapter_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(adapter_dir).expanduser().resolve() / "metadata.json").read_text(encoding="utf-8"))


def run_proposer_inference(
    rows: list[dict[str, Any]],
    prompt_template: str,
    confidence_map: dict[str, float],
    adapter_dir: str | Path,
    max_new_tokens: int = 128,
    method: str = "proposer_lora",
    model_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle = load_role_bundle(adapter_dir, model_config=model_config)
    metadata = dict(bundle["metadata"])
    predictions: list[dict[str, Any]] = []
    parse_failures = 0
    for row in rows:
        prompt_text = render_proposer_prompt(row, prompt_template)
        raw_output = generate_role_output(bundle, prompt_text, max_new_tokens=max_new_tokens)
        parsed = parse_proposer_output(raw_output, confidence_map)
        try:
            parsed = require_valid_parsed_output(row["pair_id"], parsed)
        except ValueError:
            parse_failures += 1
            raise
        predictions.append(
            {
                "pair_id": row["pair_id"],
                "repo": row["repo"],
                "window": row["window"],
                "split": row["split"],
                "gold_label": row["label"],
                "predicted_label": parsed["predicted_label"],
                "confidence": parsed["confidence"],
                "method": method,
                "issue_i": row["issue_i"],
                "issue_j": row["issue_j"],
                "raw_output": raw_output,
                "parsed_output": parsed["parsed_output"],
                "error": parsed["error"],
            }
        )
    metadata["prediction_rows"] = len(predictions)
    metadata["parse_failures"] = parse_failures
    return predictions, metadata


def train_proposer_adapter(
    examples: list[dict[str, Any]],
    model_config: dict[str, Any],
    adapter_dir: str | Path,
    max_steps: int | None = None,
    resume_from_checkpoint: str | Path | None = "auto",
    checkpoint_save_steps: int = 25,
    checkpoint_save_total_limit: int = 3,
):
    return train_role_adapter(
        examples,
        model_config,
        adapter_dir,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_save_steps=checkpoint_save_steps,
        checkpoint_save_total_limit=checkpoint_save_total_limit,
    )


def load_role_bundle(adapter_dir: str | Path, model_config: dict[str, Any] | None = None) -> dict[str, Any]:
    stack = _load_transformer_stack()
    auto_model = stack["AutoModelForCausalLM"]
    auto_tokenizer = stack["AutoTokenizer"]
    peft_model = stack["PeftModel"]

    adapter_path = Path(adapter_dir).expanduser().resolve()
    metadata = load_adapter_metadata(adapter_path)
    chosen_model = str(metadata["chosen_model_name"])
    offline = _offline_kwargs()
    tokenizer = auto_tokenizer.from_pretrained(adapter_path, **offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = _build_inference_model_kwargs(model_config)
    model_kwargs.update(offline)
    model = auto_model.from_pretrained(chosen_model, **revision_kwargs(chosen_model), **model_kwargs)
    model = peft_model.from_pretrained(model, adapter_path)
    device = _prepare_device(model_config) if model_config and model_config.get("device_preference") else select_inference_device(metadata.get("device"))
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()
    return {
        "adapter_path": adapter_path,
        "metadata": metadata,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "max_seq_length": int(metadata.get("max_seq_length") or 512),
    }


def load_multi_role_bundle(
    adapter_dirs: dict[str, str | Path],
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not adapter_dirs:
        raise ValueError("adapter_dirs must contain at least one role")

    stack = _load_transformer_stack()
    auto_model = stack["AutoModelForCausalLM"]
    auto_tokenizer = stack["AutoTokenizer"]
    peft_model = stack["PeftModel"]

    resolved_adapters = {
        str(role): Path(adapter_dir).expanduser().resolve()
        for role, adapter_dir in adapter_dirs.items()
    }
    first_role = next(iter(resolved_adapters))
    first_adapter = resolved_adapters[first_role]
    metadata_by_role = {
        role: load_adapter_metadata(adapter_path)
        for role, adapter_path in resolved_adapters.items()
    }
    first_metadata = metadata_by_role[first_role]
    chosen_model = str(first_metadata["chosen_model_name"])
    offline = _offline_kwargs()

    mismatched_models = {
        role: metadata.get("chosen_model_name")
        for role, metadata in metadata_by_role.items()
        if str(metadata.get("chosen_model_name")) != chosen_model
    }
    if mismatched_models:
        raise ValueError(f"all adapters must share one base model; mismatches: {mismatched_models}")

    tokenizer = auto_tokenizer.from_pretrained(first_adapter, **offline)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = _build_inference_model_kwargs(model_config)
    model_kwargs.update(offline)
    base_model = auto_model.from_pretrained(chosen_model, **revision_kwargs(chosen_model), **model_kwargs)
    model = peft_model.from_pretrained(base_model, first_adapter, adapter_name=first_role)
    for role, adapter_path in resolved_adapters.items():
        if role == first_role:
            continue
        model.load_adapter(adapter_path, adapter_name=role)

    device = _prepare_device(model_config) if model_config and model_config.get("device_preference") else select_inference_device(first_metadata.get("device"))
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()

    return {
        "adapter_paths": resolved_adapters,
        "metadata_by_role": metadata_by_role,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "max_seq_length": int(first_metadata.get("max_seq_length") or 512),
        "roles": list(resolved_adapters),
        "active_adapter": first_role,
    }


def select_role_bundle(bundle: dict[str, Any], role: str) -> dict[str, Any]:
    role_name = str(role)
    adapter_paths = bundle.get("adapter_paths") or {}
    if role_name not in adapter_paths:
        raise KeyError(f"unknown adapter role: {role_name}")
    selected = dict(bundle)
    selected["active_adapter"] = role_name
    selected["adapter_path"] = adapter_paths[role_name]
    selected["metadata"] = (bundle.get("metadata_by_role") or {}).get(role_name, {})
    return selected


def generate_role_output(bundle: dict[str, Any], prompt_text: str, max_new_tokens: int = 128) -> str:
    import torch

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    active_adapter = bundle.get("active_adapter")
    if active_adapter and hasattr(model, "set_adapter"):
        model.set_adapter(str(active_adapter))
    device = bundle["device"]
    from .context import fit_prompt
    prompt_text = fit_prompt(prompt_text, tokenizer, int(bundle["max_seq_length"]) - max_new_tokens)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_role_outputs(
    bundle: dict[str, Any],
    prompt_texts: list[str],
    max_new_tokens: int = 128,
    batch_size: int | None = None,
) -> list[str]:
    """Generate outputs for multiple prompts with the same role adapter.

    This preserves the single-prompt generation semantics while allowing same-role
    prompts to share one tokenizer/model generate call per chunk.
    """
    if not prompt_texts:
        return []
    if batch_size is None or batch_size <= 0:
        batch_size = len(prompt_texts)

    import torch

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    active_adapter = bundle.get("active_adapter")
    if active_adapter and hasattr(model, "set_adapter"):
        model.set_adapter(str(active_adapter))
    device = bundle["device"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = getattr(model, "config", None)
    if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
        model_config.pad_token_id = tokenizer.pad_token_id
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"

    outputs: list[str] = []
    try:
        for start in range(0, len(prompt_texts), batch_size):
            from .context import fit_prompt
            batch_prompts = [fit_prompt(p, tokenizer, int(bundle["max_seq_length"]) - max_new_tokens) for p in prompt_texts[start:start + batch_size]]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_width = inputs["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row_index in range(len(batch_prompts)):
                generated_tokens = generated[row_index][input_width:]
                outputs.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    return outputs
