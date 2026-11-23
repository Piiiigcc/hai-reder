from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "run_tfidf_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_tfidf_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_tfidf_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_tfidf_baseline)


def make_row(pair_id: str, label: str, split: str, text_suffix: str) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": {
            "train": "W1",
            "online": "W4",
            "validation": "W3",
            "test": "W7",
        }[split],
        "split": split,
        "pair_available_at": "2024-01-01T00:00:00+00:00",
        "issue_i": {
            "id": f"{pair_id}-A",
            "project": "AP",
            "created": "2024-01-01T00:00:00+00:00",
            "text": f"Requirement A {text_suffix}",
        },
        "issue_j": {
            "id": f"{pair_id}-B",
            "project": "AP",
            "created": "2024-01-01T00:00:00+00:00",
            "text": f"Requirement B {text_suffix}",
        },
        "label": label,
        "dependency_type": label,
        "dependency_types": [label],
        "raw_link_type": label,
        "raw_link_types": [label],
        "is_positive": label != "None",
        "source": "test_fixture",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_run_tfidf_baseline_trains_on_train_and_online_splits(tmp_path, monkeypatch) -> None:
    train_rows = [
        make_row("train-1", "Composition", "train", "compose alpha"),
        make_row("train-2", "Composition", "train", "compose beta"),
    ]
    online_rows = [
        make_row("online-1", "Workflow", "online", "workflow alpha"),
        make_row("online-2", "Workflow", "online", "workflow beta"),
    ]
    validation_rows = [make_row("validation-1", "Workflow", "validation", "workflow validation")]
    test_rows = [make_row("test-1", "Workflow", "test", "workflow test")]

    train_path = tmp_path / "minimal_train.jsonl"
    online_path = tmp_path / "minimal_online.jsonl"
    validation_path = tmp_path / "minimal_validation.jsonl"
    test_path = tmp_path / "minimal_test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(online_path, online_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(test_path, test_rows)

    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text("run_id: test_dataset\n", encoding="utf-8")
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text("run_id: test_experiment\n", encoding="utf-8")

    metrics_out = tmp_path / "outputs" / "metrics.json"
    table_out = tmp_path / "outputs" / "table.md"
    model_out = tmp_path / "models" / "baseline.joblib"
    pred_prefix = tmp_path / "outputs" / "predictions" / "baseline"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tfidf_baseline.py",
            "--dataset-config",
            str(dataset_config),
            "--experiment-config",
            str(experiment_config),
            "--train",
            str(train_path),
            "--online",
            str(online_path),
            "--validation",
            str(validation_path),
            "--test",
            str(test_path),
            "--model-out",
            str(model_out),
            "--pred-prefix",
            str(pred_prefix),
            "--metrics-out",
            str(metrics_out),
            "--table-out",
            str(table_out),
            "--max-features",
            "32",
        ],
    )

    exit_code = run_tfidf_baseline.main()

    assert exit_code == 0
    report = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert report["training_data_splits"] == ["train", "online"]
    assert report["training_row_count"] == len(train_rows) + len(online_rows)
    assert report["training_label_counts"] == {"Composition": 2, "Workflow": 2}
    assert report["split_reports"]["online"]["rows"] == len(online_rows)
    assert report["split_reports"]["validation"]["unseen_gold_labels_from_training_data"] == []
    assert report["split_reports"]["test"]["unseen_gold_labels_from_training_data"] == []
