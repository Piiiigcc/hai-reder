from __future__ import annotations

from typing import Any

from haireder.training.lora_trainer import _load_transformer_stack, _offline_kwargs, select_inference_device
from haireder.utils.env import detect_torch_backend
from haireder.utils.model_versions import revision_kwargs

__all__ = ["LLMClient"]


_MODEL_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _resolve_device(device: str | None) -> str:
    if device is not None:
        requested = device.strip()
        if requested and requested.lower() != "auto":
            return requested

    backend = detect_torch_backend()
    if backend.get("cuda_available"):
        return "cuda"
    if backend.get("mps_available"):
        return "mps"
    return select_inference_device()


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is not None:
        return

    for candidate in (
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "unk_token", None),
    ):
        if candidate is not None:
            tokenizer.pad_token = candidate
            return


def _load_model_bundle(model_name: str, device: str | None = None) -> dict[str, Any]:
    cache_key = (model_name, _resolve_device(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    stack = _load_transformer_stack()
    auto_model = stack["AutoModelForCausalLM"]
    auto_tokenizer = stack["AutoTokenizer"]

    offline = {**_offline_kwargs(), **revision_kwargs(model_name)}
    resolved_device = _resolve_device(device)
    tokenizer = auto_tokenizer.from_pretrained(model_name, **offline)
    _ensure_pad_token(tokenizer)

    model = auto_model.from_pretrained(model_name, torch_dtype="auto", **offline)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(resolved_device)
    model.eval()

    bundle = {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "model": model,
        "device": resolved_device,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


class LLMClient:
    """Shared LLM inference client for augmentation and other non-LoRA inference tasks."""

    __slots__ = ("_model_name", "_max_seq_length", "_bundle")

    def __init__(self, model_name: str, device: str | None = None, max_seq_length: int = 2048):
        self._model_name = str(model_name)
        self._max_seq_length = int(max_seq_length)
        if self._max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")
        self._bundle = _load_model_bundle(self._model_name, device=device)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0 when sampling is enabled")

        import torch

        tokenizer = self._bundle["tokenizer"]
        model = self._bundle["model"]
        device = self._bundle["device"]

        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] + max_new_tokens > self._max_seq_length:
            raise ValueError(f"Generation prompt exceeds the {self._max_seq_length}-token budget")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[0][prompt_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    @property
    def model_name(self) -> str:
        return self._model_name


class APILLMClient:

    __slots__ = ("_api_key", "_model", "_base_url", "_max_seq_length", "_timeout", "_client", "_session")

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        max_seq_length: int = 8192,
        timeout: int = 120,
    ):
        self._api_key = str(api_key).strip()
        if not self._api_key:
            raise ValueError("api_key must not be empty")

        self._model = str(model)
        self._base_url = str(base_url).strip().rstrip("/") or "https://api.openai.com/v1"
        self._max_seq_length = int(max_seq_length)
        if self._max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")
        self._timeout = int(timeout)
        if self._timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        self._client = None
        self._session = None

        openai_client_cls = None
        try:
            from openai import OpenAI
        except Exception:
            pass
        else:
            openai_client_cls = OpenAI

        if openai_client_cls is not None:
            try:
                self._client = openai_client_cls(api_key=self._api_key, base_url=self._base_url)
            except Exception:
                self._client = None

        if self._client is None:
            try:
                import requests
            except Exception as exc:
                raise ImportError(
                    "OpenAI SDK is unavailable and requests fallback could not be imported"
                ) from exc
            self._session = requests.Session()

    @staticmethod
    def _extract_content_from_response(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            raise RuntimeError("API response did not include any choices")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is None:
            raise RuntimeError("API response did not include message content")

        if isinstance(content, list):
            pieces: list[str] = []
            for piece in content:
                if isinstance(piece, dict):
                    text = piece.get("text")
                    if text is not None:
                        pieces.append(str(text))
                        continue
                pieces.append(str(piece))
            content = "".join(pieces)

        return str(content).strip()

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        text = f"{exc.__class__.__name__}: {exc}".lower()
        non_retry_markers = (
            "authentication",
            "unauthorized",
            "forbidden",
            "bad request",
            "unprocessable",
            "not found",
            "400",
            "401",
            "403",
            "404",
            "422",
        )
        if any(marker in text for marker in non_retry_markers):
            return False
        retry_markers = (
            "timeout",
            "timed out",
            "rate limit",
            "too many requests",
            "connection",
            "service unavailable",
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
            "temporarily overloaded",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        if any(marker in text for marker in retry_markers):
            return True
        return True

    def _generate_with_requests(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        if self._session is None:
            raise RuntimeError("requests fallback is not available")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = self._session.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            error_message = (response.text or "").strip() or "unknown error"
            raise RuntimeError(
                f"API request failed with status {response.status_code}: {error_message}"
            )
        return self._extract_content_from_response(response.json())

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        import time

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                if self._client is not None:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return self._extract_content_from_response(response)
                return self._generate_with_requests(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= 2 or not self._should_retry(exc):
                    break
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to generate text from remote LLM after 3 attempts: {last_error}") from last_error

    @property
    def model_name(self) -> str:
        return self._model


class MLXClient:
    """Keep one active model in unified memory, including when alternating validation and generation."""
    _active_name = None
    _active_bundle = None

    def __init__(self, model_name):
        self.model_name = model_name

    def bundle(self):
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        import gc
        import mlx.core as mx
        if MLXClient._active_name != self.model_name:
            MLXClient._active_name = None
            MLXClient._active_bundle = None
            gc.collect()
            mx.clear_cache()
            path = self.model_name
            if revision_kwargs(path):
                path = snapshot_download(path, **revision_kwargs(path), allow_patterns=["*.safetensors","*.json","*.txt","*.jinja"])
            MLXClient._active_bundle = load(path)
            MLXClient._active_name = self.model_name
        return MLXClient._active_bundle

    def generate_many(self, prompts, max_new_tokens=256, temperature=0.0):
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = self.bundle()
        ids = [tokenizer.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True,
                                              tokenize=True) for p in prompts]
        return batch_generate(model,tokenizer,ids,max_tokens=max_new_tokens,
                              sampler=make_sampler(temp=temperature),verbose=False).texts

    def generate(self, prompt, max_new_tokens=256, temperature=0.0):
        return self.generate_many([prompt], max_new_tokens, temperature)[0]


def create_llm_client(
    backend: str = "local",
    **kwargs,
) -> LLMClient | APILLMClient:
    if not isinstance(backend, str):
        config = dict(backend)
        config.update(kwargs)
        backend = str(config.pop("backend", "local"))
        kwargs = config

    backend = backend.lower()
    if backend == "mlx":
        return MLXClient(kwargs["model_name"])
    if backend == "api":
        return APILLMClient(
            api_key=kwargs["api_key"],
            model=kwargs.get("model", "gpt-4o-mini"),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            max_seq_length=kwargs.get("max_seq_length", 8192),
            timeout=kwargs.get("timeout", 120),
        )
    if backend == "local":
        return LLMClient(
            model_name=kwargs["model_name"],
            device=kwargs.get("device"),
            max_seq_length=kwargs.get("max_seq_length", 8192),
        )
    raise ValueError("backend must be 'local' or 'api'")


__all__ = [*__all__, "APILLMClient", "create_llm_client"]
