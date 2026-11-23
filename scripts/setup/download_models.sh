#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src python - <<'PY'
from huggingface_hub import snapshot_download
from haireder.utils.model_versions import revision_kwargs
for name in ("Qwen/Qwen2.5-7B-Instruct","Qwen/Qwen2.5-3B-Instruct",
             "sentence-transformers/all-MiniLM-L6-v2","google-bert/bert-base-uncased"):
    print(snapshot_download(name,**revision_kwargs(name),allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"],
                            ignore_patterns=["onnx/*","openvino/*"]),flush=True)
PY
