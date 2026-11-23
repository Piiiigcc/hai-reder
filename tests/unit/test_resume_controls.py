from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.training.lora_trainer import (  # noqa: E402
    generate_role_output,
    latest_trainer_checkpoint,
    load_multi_role_bundle,
    resolve_resume_checkpoint,
    select_role_bundle,
)


def test_latest_trainer_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    (adapter_dir / "trainer_state" / "checkpoint-2").mkdir(parents=True)
    (adapter_dir / "trainer_state" / "checkpoint-10").mkdir(parents=True)
    (adapter_dir / "trainer_state" / "not-a-checkpoint").mkdir()

    latest = latest_trainer_checkpoint(adapter_dir)

    assert latest == adapter_dir.resolve() / "trainer_state" / "checkpoint-10"


def test_resolve_resume_checkpoint_supports_auto_none_and_explicit_path(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    explicit = tmp_path / "manual-checkpoint"
    explicit.mkdir()
    (adapter_dir / "trainer_state" / "checkpoint-4").mkdir(parents=True)

    assert resolve_resume_checkpoint(adapter_dir, "auto") == adapter_dir.resolve() / "trainer_state" / "checkpoint-4"
    assert resolve_resume_checkpoint(adapter_dir, "none") is None
    assert resolve_resume_checkpoint(adapter_dir, explicit) == explicit.resolve()


def test_train_role_adapter_passes_4bit_quantization_config_when_enabled(monkeypatch, tmp_path: Path) -> None:
    import types

    from haireder.training.lora_trainer import train_role_adapter

    class FakeDType:
        def __init__(self, name: str) -> None:
            self.name = name

        def __str__(self) -> str:
            return f"torch.{self.name}"

        __repr__ = __str__

    class BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = BitsAndBytesConfig
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = FakeDType("float16")
    fake_torch.bfloat16 = FakeDType("bfloat16")
    fake_torch.float32 = FakeDType("float32")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.saved_paths: list[Path] = []

        def save_pretrained(self, path: str | Path) -> None:
            self.saved_paths.append(Path(path))

    class DummyAutoTokenizer:
        def __init__(self, tokenizer: DummyTokenizer) -> None:
            self.tokenizer = tokenizer
            self.calls: list[str] = []

        def from_pretrained(self, model_name: str, *, local_files_only: bool = False, revision=None):
            from haireder.utils.model_versions import REVISIONS
            assert revision == REVISIONS[model_name]
            self.calls.append(model_name)
            return self.tokenizer

    class DummyModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"pad_token_id": None, "use_cache": True})()
            self.saved_paths: list[Path] = []
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

        def save_pretrained(self, path: str | Path) -> None:
            self.saved_paths.append(Path(path))

    class DummyAutoModel:
        def __init__(self, model: DummyModel) -> None:
            self.model = model
            self.calls: list[dict[str, object]] = []

        def from_pretrained(self, model_name: str, **kwargs):
            self.calls.append({"model_name": model_name, "kwargs": kwargs})
            return self.model

    class DummyTrainingArguments:
        instances: list["DummyTrainingArguments"] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.__class__.instances.append(self)

    class DummyTrainer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.model = kwargs["model"]

        def train(self, resume_from_checkpoint=None):
            self.resume_from_checkpoint = resume_from_checkpoint
            return type("Result", (), {"metrics": {"train_loss": 0.0}})()

    class DummyLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DummyTaskType:
        CAUSAL_LM = "causal_lm"

    import peft
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training",lambda model, **kwargs: model)
    tokenizer = DummyTokenizer()
    model = DummyModel()
    auto_tokenizer = DummyAutoTokenizer(tokenizer)
    auto_model = DummyAutoModel(model)

    monkeypatch.setattr(
        "haireder.training.lora_trainer._load_transformer_stack",
        lambda: {
            "Dataset": object,
            "LoraConfig": DummyLoraConfig,
            "PeftModel": object,
            "TaskType": DummyTaskType,
            "get_peft_model": lambda trained_model, _lora_config: trained_model,
            "AutoModelForCausalLM": auto_model,
            "AutoTokenizer": auto_tokenizer,
            "Trainer": DummyTrainer,
            "TrainingArguments": DummyTrainingArguments,
            "default_data_collator": lambda batch: batch,
        },
    )
    monkeypatch.setattr("haireder.training.lora_trainer.choose_base_model_name", lambda model_config: "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setattr("haireder.training.lora_trainer._prepare_device", lambda model_config: "cpu")
    monkeypatch.setattr(
        "haireder.training.lora_trainer._tokenize_examples",
        lambda examples, tokenizer, max_length: [{"input_ids": [1], "attention_mask": [1], "labels": [1]}],
    )

    model_config = {
        "base_model": {"primary": "Qwen/Qwen2.5-7B-Instruct"},
        "device_preference": ["cuda", "cpu"],
        "tokenization": {"max_seq_length": 16},
        "quantization": {
            "enabled": True,
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": "auto"},
        "training": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "epochs": 1,
            "seed": 42,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
        },
    }

    train_role_adapter([], model_config, tmp_path / "adapter", max_steps=1, resume_from_checkpoint=None)

    assert auto_model.calls, "expected model.from_pretrained to be called"
    from haireder.utils.model_versions import REVISIONS
    assert auto_model.calls[0]["kwargs"]["revision"] == REVISIONS["Qwen/Qwen2.5-7B-Instruct"]
    quantization_config = auto_model.calls[0]["kwargs"].get("quantization_config")
    assert quantization_config is not None
    assert quantization_config.__class__.__name__ == "BitsAndBytesConfig"
    assert quantization_config.load_in_4bit is True
    assert str(quantization_config.bnb_4bit_compute_dtype) == "torch.bfloat16"
    assert quantization_config.bnb_4bit_quant_type == "nf4"
    assert quantization_config.bnb_4bit_use_double_quant is True
    assert model.gradient_checkpointing_enabled is True
    assert model.config.use_cache is False
    assert DummyTrainingArguments.instances[-1].kwargs["bf16"] is True
    assert DummyTrainingArguments.instances[-1].kwargs["fp16"] is False
    assert DummyTrainingArguments.instances[-1].kwargs["warmup_ratio"] == 0.03
    assert DummyTrainingArguments.instances[-1].kwargs["lr_scheduler_type"] == "cosine"


def test_load_multi_role_bundle_loads_one_base_and_switches_adapters(monkeypatch, tmp_path: Path) -> None:
    import json
    import types

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    class BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = BitsAndBytesConfig
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch.no_grad = lambda: NoGrad()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class DummyTensor:
        shape = (1, 2)

        def to(self, device):
            return self

    class DummyTokenizer:
        pad_token = "</s>"
        eos_token = "</s>"
        pad_token_id = 0

        def encode(self,text,add_special_tokens=False):
            return [1,2]

        def __call__(self, *args, **kwargs):
            return {"input_ids": DummyTensor(), "attention_mask": DummyTensor()}

        def decode(self, tokens, skip_special_tokens=True):
            return "{}"

    class DummyAutoTokenizer:
        def from_pretrained(self, path, *, local_files_only: bool = False):
            assert local_files_only
            return DummyTokenizer()

    class DummyModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"pad_token_id": None})()
            self.loaded_adapters: list[tuple[str, str]] = []
            self.active_adapters: list[str] = []
            self.to_calls: list[str] = []

        def load_adapter(self, adapter_path, adapter_name):
            self.loaded_adapters.append((str(adapter_path), adapter_name))

        def set_adapter(self, adapter_name):
            self.active_adapters.append(adapter_name)

        def to(self, device):
            self.to_calls.append(device)
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class DummyAutoModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.model = DummyModel()

        def from_pretrained(self, model_name, **kwargs):
            self.calls.append({"model_name": model_name, "kwargs": kwargs})
            return self.model

    class DummyPeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path, adapter_name=None):
            base_model.loaded_adapters.append((str(adapter_path), adapter_name or "default"))
            return base_model

    auto_model = DummyAutoModel()
    monkeypatch.setattr(
        "haireder.training.lora_trainer._load_transformer_stack",
        lambda: {
            "Dataset": object,
            "LoraConfig": object,
            "PeftModel": DummyPeftModel,
            "TaskType": object,
            "get_peft_model": object,
            "AutoModelForCausalLM": auto_model,
            "AutoTokenizer": DummyAutoTokenizer(),
            "Trainer": object,
            "TrainingArguments": object,
            "default_data_collator": object,
        },
    )
    monkeypatch.setattr("haireder.training.lora_trainer.select_inference_device", lambda metadata_device=None: "cuda")

    adapter_dirs = {}
    for role in ("proposer", "critic", "judge"):
        adapter_dir = tmp_path / role
        adapter_dir.mkdir()
        (adapter_dir / "metadata.json").write_text(
            json.dumps({"chosen_model_name": "base-model", "device": "cuda", "max_seq_length": 16}),
            encoding="utf-8",
        )
        adapter_dirs[role] = adapter_dir

    model_config = {
        "quantization": {
            "enabled": True,
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    }

    bundle = load_multi_role_bundle(adapter_dirs, model_config=model_config)
    critic_bundle = select_role_bundle(bundle, "critic")
    generate_role_output(critic_bundle, "prompt", max_new_tokens=4)

    assert len(auto_model.calls) == 1
    assert auto_model.calls[0]["kwargs"]["local_files_only"] is True
    assert auto_model.calls[0]["kwargs"]["quantization_config"].load_in_4bit is True
    assert auto_model.model.loaded_adapters == [
        (str(adapter_dirs["proposer"].resolve()), "proposer"),
        (str(adapter_dirs["critic"].resolve()), "critic"),
        (str(adapter_dirs["judge"].resolve()), "judge"),
    ]
    assert auto_model.model.active_adapters[-1] == "critic"
