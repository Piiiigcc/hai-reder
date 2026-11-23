# HAI-ReDeR

HAI-ReDeR classifies dependency relationships between ordered pairs of Jira requirements. It combines a Proposer, Critic and Judge with role-specific LoRA adapters, confidence calibration, analyst review, and periodic updates. The code also includes TF-IDF, BERT and zero-shot baselines.

## Quick start

Use Python 3.11 from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/workflow/run_formal_pipeline.py --suite grid --dry-run
python -m pytest -q
```

The grid command prints the 64 planned configurations. Some tests load the public MiniLM encoder from Hugging Face or an existing cache.

## Run the formal pipeline

Install the CUDA dependencies on a suitable NVIDIA host using `requirements_server.txt`. Provide the Public Jira source files specified in [the dataset configuration](configs/dataset/formal_6repos.yaml). The data preparation utilities are in `scripts/data/`. The default experiment uses Qwen2.5-7B-Instruct with 4-bit NF4 QLoRA.

```bash
bash scripts/setup/download_models.sh
python scripts/data/build_creation_text.py
python scripts/data/build_formal_dataset.py
python scripts/workflow/run_formal_pipeline.py --suite full
```

The pipeline pauses when an analyst needs to review a batch. Fill the decision fields in the generated `review.csv` and resume the same configuration with `--resume`. Use `--suite ablations` for the three ablations and `--suite grid` for sensitivity runs.

The classifier and training logic are in `src/haireder/`. Prompts and experiment settings are in `prompts/` and `configs/`. The label mapping is stored with the taxonomy configuration.
