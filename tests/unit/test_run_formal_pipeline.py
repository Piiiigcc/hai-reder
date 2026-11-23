"""Behavioral tests for the paper entry point, cohort guards and resumable human review."""

from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
from haireder.experiments.evaluation import (
    evaluation,
    fit_bundle,
    calibrate,
    validate_predictions,
)
from haireder.experiments.review import review_queue, read_decisions, PendingReviews
from haireder.experiments.runner import plans
from haireder.experiments.sprint import process_sprint
from haireder.experiments.stages import Stages
from haireder.utils.config import load_yaml
from haireder.confidence.signals import compute_inter_agent_agreement_signal

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/workflow/run_formal_pipeline.py"


def item(key, region="low", window="W5"):
    return dict(
        pair_id=key,
        repo="R",
        window=window,
        split="online",
        issue_i=dict(id="A" + key, text="First " + key),
        issue_j=dict(id="B" + key, text="Second " + key),
        confidence_region=region,
        confidence={"low": 0.2, "medium": 0.6, "high": 0.9}[region],
        predicted_label="Composition",
        transcript={
            "rounds": [],
            "judge": {"parsed_output": {"justification": "argument"}},
        },
    )


def fill(path, values, *, reviewer="test_fixture"):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    for r in rows:
        if r["pair_id"] in values:
            r.update(
                accepted_label=values[r["pair_id"]],
                reviewer=reviewer,
                reviewed_at="2026-11-22T12:00:00Z",
                rationale="Controlled review fixture",
                preserves_relation="yes" if r["kind"] == "augmentation" else "",
            )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_help_documents_resume_and_handoffs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in (
        "--suite",
        "--checkpoint-path",
        "--from-stage",
        "--force",
        "--resume",
        "--log-dir",
        "--dry-run",
        "--timeout",
    ):
        assert flag in result.stdout


def test_grid_schedules_64_unique_runs_without_model_or_data(tmp_path):
    path = tmp_path / "absent"
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--suite",
            "grid",
            "--dry-run",
            "--dataset-dir",
            str(path),
            "--run-dir",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(p.stdout)
    assert data["status"] == "plan_only" and len(data["configurations"]) == 64
    assert len({x["name"] for x in data["configurations"]}) == 64
    assert not path.exists()


def test_three_ablations_change_executable_settings():
    full, single, static, random = plans("ablations")
    assert full["dynamic"] and full["augmentation"] and not full["single"]
    assert single["single"] and single["dynamic"] and single["augmentation"]
    assert not static["dynamic"] and static["augmentation"] and not static["single"]
    assert random["negatives"] == "random" and not random["augmentation"]
    assert len({p["name"] for p in (full, single, static, random)}) == 4


def test_formal_settings():
    cfg = load_yaml("configs/experiments/formal_full_pipeline.yaml")
    model = load_yaml(cfg["model_config"])
    assert model["base_model"]["primary"] == "Qwen/Qwen2.5-7B-Instruct"
    assert (
        cfg["debate"]["max_rounds"] == 3
        and cfg["online_learning"]["refresh_period_sprints"] == 4
    )
    assert (
        model["training"]["gradient_checkpointing"]
        and model["training"]["max_steps"] is None
    )
    assert (
        cfg["training"]["augmentation"]["llm"]["model_name"]
        != cfg["training"]["augmentation"]["llm"]["validator_model_name"]
    )


def test_inference_quotes_must_come_from_the_correct_endpoint():
    from haireder.experiments.roles import verify_quotes

    row = item("quoted")
    verify_quotes({"quote_a": "First", "quote_b": "Second"}, row)
    for invalid in ("First", None, ".", ""):
        with pytest.raises(ValueError, match="quote_b"):
            verify_quotes({"quote_a": "First", "quote_b": invalid}, row)


def test_scoring_rejects_tagged_fallback_even_with_a_legal_final_label():
    expected = dict(item("x"), label="Composition")
    actual = dict(expected, gold_label="Composition", error=None)
    actual["transcript"] = {
        "rounds": [{"critic": {"parsed_output": {"is_fallback_generation": True}}}],
        "judge": {"parsed_output": {"final_label": "Composition"}},
    }
    with pytest.raises(ValueError, match="substituted"):
        validate_predictions([expected], [actual])


@pytest.mark.parametrize(
    "tag",
    [{"re_infer_source": "placeholder_deterministic"}, {"analyst_simulation": True}],
)
def test_diagnostic_predictions_cannot_enter_formal_score_or_reinference(tmp_path, tag):
    from haireder.experiments.runner import score
    from haireder.experiments.evaluation import INFERENCE_FIELDS

    original = item("controlled", "medium")
    source = dict(original, label="Workflow", is_positive=True)

    class Diagnostic:
        def infer(self, row, pool):
            assert set(row) == set(INFERENCE_FIELDS)
            return dict(original, **tag)

    with pytest.raises(ValueError, match="substituted"):
        score(
            Diagnostic(),
            [source],
            [],
            load_yaml("configs/experiments/formal_full_pipeline.yaml"),
            None,
            plans("full")[0],
        )
    with pytest.raises(ValueError, match="substituted"):
        process_sprint(
            [original],
            [],
            {},
            {},
            tmp_path,
            "controlled",
            lambda rows, pool: [dict(rows[0], **tag)],
        )
    assert not (tmp_path / "reinference.json").exists()


def test_formal_dataset_requires_bound_splits_and_unique_chronological_rows(tmp_path):
    from haireder.data.formal import file_hash
    from haireder.experiments.runner import dataset_files
    from haireder.utils.io import write_json, write_jsonl

    files = {}
    for split, window in {
        "train": "W1",
        "validation": "W4",
        "online": "W5",
        "test": "W29",
    }.items():
        files[split] = write_jsonl(
            tmp_path / f"formal_{split}.jsonl",
            [dict(item(split, window=window), split=split, label="None")],
        )
    manifest = dict(
        mode="formal",
        inputs={},
        repositories=["R"],
        split_counts=dict.fromkeys(files, 1),
    )
    path = tmp_path / "formal_manifest.json"

    def freeze():
        write_json(
            path,
            dict(manifest, artifacts={f.name: file_hash(f) for f in files.values()}),
        )

    freeze()
    assert len(dataset_files(tmp_path)[1]["online"]) == 1
    write_json(path, dict(manifest, artifacts={}))
    with pytest.raises(ValueError, match="lacks verified"):
        dataset_files(tmp_path)
    for row in (
        dict(item("train", window="W5"), split="online", label="None"),
        dict(item("new", window="W29"), split="online", label="None"),
    ):
        write_jsonl(files["online"], [row])
        freeze()
        with pytest.raises(ValueError, match="Invalid or duplicate"):
            dataset_files(tmp_path)


def test_training_augmentation_can_resume_after_a_whole_rejected_block(
    tmp_path, monkeypatch
):
    from haireder.augmentation import LLMPairAugmenter, LLMConsistencyValidator
    from haireder.experiments import augmentation
    from haireder.taxonomy import LABELS
    from types import SimpleNamespace

    train = [
        dict(item(f"{i}_{j}", window="W1"), split="train", label=label)
        for i, label in enumerate(LABELS)
        for j in range(1 if label == "Duplication" else 2)
    ]
    permitted = False

    def components(config, prompt, validator_prompt, taxonomy):
        def generate(prompts, **kwargs):
            return [
                json.dumps(
                    dict(
                        augmented_issue_i_text="first rewritten test requirement",
                        augmented_issue_j_text="second rewritten test requirement",
                    )
                )
                for _ in prompts
            ]

        def validate(prompts, **kwargs):
            return [
                json.dumps(
                    dict(
                        is_valid=permitted, label_matches=permitted, rationale="fixture"
                    )
                )
                for _ in prompts
            ]

        return (
            LLMPairAugmenter(SimpleNamespace(generate_many=generate), prompt, taxonomy),
            LLMConsistencyValidator(
                SimpleNamespace(generate_many=validate), validator_prompt
            ),
            SimpleNamespace(filter=lambda *args: dict(is_valid=True, distance=0.4)),
        )

    monkeypatch.setattr(augmentation, "build_llm_components", components)
    cfg = load_yaml("configs/experiments/formal_full_pipeline.yaml")
    with pytest.raises(RuntimeError, match="shortfall"):
        augmentation.generate_pairs(train, tmp_path, cfg)
    assert json.loads((tmp_path / "balance.json").read_text())["attempts"] == 32
    permitted = True
    assert len(augmentation.generate_pairs(train, tmp_path, cfg)) == 1
    summary = json.loads((tmp_path / "balance.json").read_text())
    assert summary["attempts"] == 33 and summary["missing"] == {}


def test_explicit_concession_and_budget_have_distinct_traces():
    from haireder.debate.engine import run_arguments

    def propose(r, i, transcript):
        return {"parsed_output": {"label": "Workflow"}}

    def critique(r, turn, i, transcript):
        return {
            "parsed_output": {
                "stance": "concede" if i else "challenge",
                "label": "Workflow" if i else "Composition",
            }
        }

    assert (
        run_arguments(item("c"), propose, critique, 1)["termination_reason"]
        == "round_budget"
    )
    result = run_arguments(item("c"), propose, critique, 3)
    assert (
        len(result["rounds"]) == 2 and result["termination_reason"] == "critic_concede"
    )


def test_grid_aggregation_only_compares_verified_four_cell_contrasts(tmp_path):
    from haireder.utils.io import write_json, write_jsonl
    from haireder.data.formal import file_hash

    spec = importlib.util.spec_from_file_location("paper_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    grid = plans("grid")
    selected = [
        p for p in grid if p["low"] <= 0.2 and p["high"] == 0.6 and p["period"] <= 4
    ]
    f1 = {(0.1, 2): 0.4, (0.1, 4): 0.5, (0.2, 2): 0.5, (0.2, 4): 0.7}
    for case in selected:
        d = tmp_path / case["name"]
        config = write_json(
            d / "configuration.json", dict(identity="fixture", plan=case)
        )
        f = f1[case["low"], case["period"]]
        summary = write_json(
            d / "summary.json",
            dict(
                plan=case,
                held_out=dict(macro_f1=f, ece=0.1, aurc=0.3),
                workload=dict(total_review_count=3),
                cohort="shared-fixture",
            ),
        )
        preds = write_jsonl(d / "test_predictions.jsonl", [{"pair_id": "test-fixture"}])
        st = Stages(d, "fixture")
        st.run("final_evaluation", "final_evaluation", 1, lambda: [summary, preds])
        write_json(
            d / "status.json",
            dict(
                status="complete",
                identity="fixture",
                configuration_sha256=file_hash(config),
                checkpoint_sha256=file_hash(st.path),
            ),
        )
    result = cli.aggregate(tmp_path, grid)
    assert result["completed"] == 4 and result["total"] == 64
    assert len(result["interactions"]) == 1 and result["interactions"][0]["effects"][
        "macro_f1"
    ] == pytest.approx(0.1)
    assert (
        sum(r.get("pareto_f1_review_count", False) for r in result["configurations"])
        == 1
    )
    requiring_baselines = cli.aggregate(tmp_path, grid, require_baselines=True)
    assert requiring_baselines["completed"] == 0
    incomplete = next(
        r for r in requiring_baselines["configurations"] if r["name"] == case["name"]
    )
    assert incomplete["status"] == "awaiting_baselines"
    assert set(incomplete["missing_baselines"]) == {"tfidf", "bert", "zero_shot"}
    cli.record_failure(d, status="blocked", reason="failed retry")
    assert json.loads((d / "status.json").read_text())["status"] == "complete"
    preds.write_text("{}\n")
    with pytest.raises(ValueError, match="fingerprint"):
        cli.aggregate(tmp_path, grid)
    st.path.unlink()
    with pytest.raises(ValueError, match="Unverified"):
        cli.aggregate(tmp_path, grid)


def test_judge_retries_a_decision_that_overrules_consensus(monkeypatch):
    from haireder.experiments import roles

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    system = roles.RoleSystem.__new__(roles.RoleSystem)
    system.bundle, system.verbal = {}, {"high": 0.7}
    monkeypatch.setattr(system, "prompt", lambda *args: "Read both arguments")
    monkeypatch.setattr(
        roles,
        "select_role_bundle",
        lambda *args: {"tokenizer": Tokenizer(), "max_seq_length": 512},
    )
    answers = iter(("None", "Workflow"))

    def generate(*args, **kwargs):
        return json.dumps(
            dict(
                final_label=next(answers),
                justification="Original words from both issues",
                quote_a="First",
                quote_b="Second",
                verbal_confidence="high",
            )
        )

    monkeypatch.setattr(roles, "generate_role_output", generate)
    turn = {"parsed_output": {"label": "Workflow"}}
    trace = dict(
        termination_reason="agreement", rounds=[dict(proposer=turn, critic=turn)]
    )
    result = system.turn("judge", item("bound"), trace)
    assert result["predicted_label"] == "Workflow"
    (rejected,) = result["invalid_generation_attempts"]
    assert json.loads(rejected["raw_output"])["final_label"] == "None"
    assert (
        "agreed" in rejected["error"]
        and rejected["prompt_hash"] != result["prompt_hash"]
    )


def test_rerun_cannot_commit_a_substituted_endpoint(tmp_path):
    original = item("medium", "medium")

    def swapped(rows, pool):
        return [dict(rows[0], issue_j={"id": "replacement", "text": "Replacement"})]

    with pytest.raises(ValueError, match="Invalid re-inference"):
        process_sprint(
            [original],
            [],
            {},
            dict(
                pool_capacity=3,
                recency_partition_ratio=0.6,
                difficulty_partition_ratio=0.4,
            ),
            tmp_path,
            "bound",
            swapped,
        )
    assert not (tmp_path / "reinference.json").exists()


def test_timeout_kills_a_silent_process(tmp_path):
    spec = importlib.util.spec_from_file_location("paper_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start = time.monotonic()
    assert (
        mod.run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(15)"], tmp_path / "log", 0.2
        )
        == 124
    )
    assert time.monotonic() - start < 4


def test_checkpoint_requires_earlier_artifacts_and_rejects_tampering(tmp_path):
    store = Stages(tmp_path, "identity", from_stage="final_evaluation")
    f = tmp_path / "weights"

    def materialize():
        f.write_text("trained")
        return [f]

    with pytest.raises(ValueError, match="requires"):
        store.run("initial", "training", 1, materialize)
    stages = Stages(tmp_path, "identity")
    first = stages.run("initial", "training", 1, materialize)
    assert (
        Stages(tmp_path, "identity", from_stage="final_evaluation").run(
            "initial",
            "training",
            1,
            lambda: pytest.fail("completed training was run twice"),
        )
        == first
    )
    f.write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        stages.run("initial", "training", 1, materialize)
    with pytest.raises(ValueError, match="changed"):
        Stages(tmp_path, "other")


def test_failed_stage_is_uncommitted(tmp_path):
    stages = Stages(tmp_path, "id")
    with pytest.raises(ValueError):
        stages.run("initial", "training", 1, lambda: [tmp_path / "missing"])
    assert not stages.path.exists()


def test_failed_stage_compute_is_counted_after_a_retry(tmp_path, monkeypatch):
    from haireder.experiments import stages as module

    clock = iter([2.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    stages = Stages(tmp_path, "identity")

    def shortfall():
        raise RuntimeError("recoverable augmentation shortfall")

    with pytest.raises(RuntimeError, match="shortfall"):
        stages.run("augmentation", "training", 1, shortfall)
    assert not stages.path.exists()
    output = tmp_path / "qualified.jsonl"
    output.write_text("{}\n")
    stages.run("augmentation", "training", 1, lambda: [output])
    assert stages.state["stages"]["augmentation"]["elapsed_seconds"] == 13.0


def test_sprint_handoff_reinference_promotion_and_no_gold_oracle(tmp_path):
    rows = [
        dict(item("a"), gold_label="None"),
        item("b", "medium"),
        item("c", "medium"),
        item("d", "high"),
    ]
    cfg = dict(
        pool_capacity=30, recency_partition_ratio=0.6, difficulty_partition_ratio=0.4
    )

    def reinfer(medium, context):
        assert (
            context[-1]["pair_id"] == "a"
            and context[-1]["accepted_label"] == "Workflow"
        )
        return [
            dict(
                r,
                confidence=0.9 if r["pair_id"] == "b" else 0.5,
                confidence_region="high" if r["pair_id"] == "b" else "medium",
            )
            for r in medium
        ]

    with pytest.raises(PendingReviews) as pending:
        process_sprint(rows, [], {}, cfg, tmp_path, "run", reinfer)
    low_path = pending.value.path
    assert "/low/" in str(low_path)
    shown = json.loads((low_path.parent / "items.jsonl").read_text())
    assert "gold_label" not in shown and "proposed_label" not in shown
    assert shown["model_rationale"] == "Judge: argument"
    fill(low_path, {"a": "Workflow"})
    with pytest.raises(PendingReviews) as medium:
        process_sprint(rows, [], {}, cfg, tmp_path, "run", reinfer)
    assert "/medium/" in str(medium.value.path)
    fill(medium.value.path, {"c": "Duplication"})
    accepted, pool = process_sprint(rows, [], {}, cfg, tmp_path, "run", reinfer)
    assert {r["pair_id"]: r["accepted_label"] for r in accepted} == {
        "a": "Workflow",
        "b": "Composition",
        "c": "Duplication",
        "d": "Composition",
    }
    assert len(pool) == 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (
        summary["total_review_count"] == 2
        and summary["initial_medium_count"] == 2
        and summary["triage_saving"] == 0.5
    )


def test_future_evidence_rejected(tmp_path):
    with pytest.raises(ValueError, match="future"):
        process_sprint(
            [item("a")], [item("b", window="W6")], {}, {}, tmp_path, "batch", None
        )


def test_review_batch_cannot_change_after_partial_fill(tmp_path):
    candidate = item("a")
    path = review_queue(tmp_path, "b", "low", [candidate])
    fill(path, {"a": "Workflow"})
    assert read_decisions(path)["a"]["accepted_label"] == "Workflow"
    with pytest.raises(ValueError, match="changed"):
        review_queue(
            tmp_path, "b", "low", [dict(candidate, issue_i={"text": "changed"})]
        )


def test_low_review_shows_agent_reasons_and_preserves_separate_intake_judgment(
    tmp_path,
):
    from haireder.experiments.review import import_independent_decisions
    from haireder.utils.io import read_jsonl

    candidate = item("a")
    candidate["transcript"]["rounds"] = [
        {
            "proposer": {
                "parsed_output": {"label": "Workflow", "rationale": "proposal evidence"}
            },
            "critic": {
                "parsed_output": {
                    "label": "Composition",
                    "rationale": "critique evidence",
                }
            },
        }
    ]
    intake = review_queue(tmp_path / "intake", "pre", "online_intake", [candidate])
    low = review_queue(tmp_path / "low", "post", "low", [candidate])
    fill(intake, {"a": "Duplication"})
    assert import_independent_decisions(intake, low) == 1
    shown = read_jsonl(low.parent / "items.jsonl")[0]
    assert all(
        s in shown["model_rationale"]
        for s in ("proposal evidence", "critique evidence", "Judge: argument")
    )
    assert all(
        k not in shown
        for k in ("proposed_label", "predicted_label", "gold_label", "confidence")
    )
    notes = json.loads((low.parent / "intake_import.json").read_text())["notes"]
    assert notes[0]["intake"]["accepted_label"] == "Duplication"
    assert notes[0]["low_item_hash"] == shown["item_hash"]
    with pytest.raises(PendingReviews):
        read_decisions(low)
    fill(low, {"a": "Workflow"})
    assert read_decisions(low)["a"]["accepted_label"] == "Workflow"


def test_augmentation_blind_sample_preserves_seed_references(tmp_path):
    data = [dict(item(str(i)), source_seed_pair_id="seed" + str(i)) for i in range(112)]
    seeds = {"seed" + str(i): item("seed" + str(i)) for i in range(112)}
    path = review_queue(
        tmp_path, "b", "augmentation", data, seeds=seeds, sample_size=100, seed=42
    )
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len({r["pair_id"] for r in rows}) == 100 and all(
        r["seed_i"] and r["accepted_label"] == "" for r in rows
    )
    assert read_decisions(path, require_all=False) == {}
    from haireder.experiments.runner import artifact_files

    source = tmp_path / "batch"
    sheet = review_queue(
        source / "human_review",
        "bound",
        "augmentation",
        data,
        seeds=seeds,
        sample_size=100,
        seed=42,
    )
    frozen = set(artifact_files(source))
    assert frozen == {sheet.parent / "items.jsonl", sheet.parent / "batch.json"}
    with pytest.raises(ValueError, match="Need 113"):
        review_queue(tmp_path, "b", "augmentation", data, seeds=seeds, sample_size=113)


def test_joint_round_agreement_counts_both_roles():
    def r(i, p, c):
        return {
            "round_index": i,
            "proposer": {"parsed_output": {"label": p}},
            "critic": {"parsed_output": {"label": c}},
        }

    t = {
        "judge": {"parsed_output": {"final_label": "None"}},
        "rounds": [r(0, "None", "Workflow"), r(1, "None", "None"), r(2, None, "None")],
    }
    assert compute_inter_agent_agreement_signal(t)[0] == pytest.approx(1 / 3)
    assert compute_inter_agent_agreement_signal(t, round_weighted=True)[
        0
    ] == pytest.approx(2 / 6)


def test_calibration_recomputes_raw_with_selected_w4_weights():
    rows = [
        dict(
            item(str(i)),
            split="validation",
            window="W4",
            confidence=0.23,
            raw_confidence=0.23,
            s1=i % 2,
            s2=0.0,
            s3=0.5,
            gold_label="Composition" if i % 2 else "Workflow",
        )
        for i in range(24)
    ]
    bundle = fit_bundle(rows)
    high = calibrate(rows[1], bundle, 0.4, 0.8)
    low = calibrate(rows[0], bundle, 0.4, 0.8)
    assert (
        high["raw_confidence"] != 0.23
        and high["confidence"] > low["confidence"]
        and sum(bundle["weights"].values()) == pytest.approx(1)
    )
    with pytest.raises(ValueError, match="W4"):
        fit_bundle([dict(rows[0], window="W5")])
    with pytest.raises(ValueError, match="Platt"):
        fit_bundle([rows[0]])
    with pytest.raises(ValueError, match="confidence signals"):
        calibrate(dict(rows[0], s3=float("nan")), bundle, 0.4, 0.8)


@pytest.mark.parametrize(
    "kind", ["empty", "duplicate", "foreign", "label", "side", "nan"]
)
def test_final_coverage_rejects_broken_results(kind):
    r = dict(item("a"), label="Composition")
    p = dict(r, gold_label="Composition")
    predictions = {
        "empty": [],
        "duplicate": [p, p],
        "foreign": [dict(p, pair_id="other")],
        "label": [dict(p, predicted_label="Maybe")],
        "side": [dict(p, issue_j={"text": "replaced"})],
        "nan": [dict(p, confidence=float("nan"))],
    }[kind]
    with pytest.raises(ValueError):
        validate_predictions([r], predictions)


def test_fixed_macro_aurc_and_regions():
    a = dict(item("a", "high"), gold_label="Composition")
    b = dict(item("b", "low"), gold_label="Workflow")
    m = evaluation([a, b])
    assert m["macro_f1"] == pytest.approx((2 / 3) / 6)
    assert (
        m["aurc"] == pytest.approx(0.25)
        and m["regions"]["low"]["accuracy"] == 0
        and m["ece"] is not None
    )


def test_grounded_target_requires_quotes_in_both_issues():
    from haireder.experiments.roles import grounded_target

    row = dict(item("a"), label="Composition")

    class Teacher:
        model_name = "test-double"

        def generate(self, *a, **kw):
            return json.dumps(self.answer)

    teacher = Teacher()
    teacher.answer = {
        "quote_a": "First",
        "quote_b": "Second",
        "explanation": "part relation",
        "verbal_confidence": "medium",
    }
    target, evidence = grounded_target(teacher, row, "critic", {"label": "Workflow"})
    assert target["stance"] == "challenge" and evidence["quote_a"] == "First"
    teacher.answer["quote_b"] = "invented excerpt"
    with pytest.raises(ValueError, match="ground") as error:
        grounded_target(teacher, row, "proposer")
    assert str(error.value).count("invented excerpt") == 3


def test_grounded_target_retry_requires_side_specific_excerpt_ids():
    from haireder.experiments.roles import grounded_target

    reason = dict(
        explanation="The two titles support the label", verbal_confidence="high"
    )
    answers = iter(
        [
            dict(reason, quote_a="Second a", quote_b="First a"),
            dict(reason, quote_a_id="B1", quote_b_id="A1"),
            dict(reason, quote_a_id="A1", quote_b_id="B1"),
        ]
    )

    class Teacher:
        model_name = "test-double"

        def generate(self, prompt, **kwargs):
            answer = next(answers)
            if "quote_a_id" in answer:
                assert '"A1": "First a"' in prompt
                assert '"B1": "Second a"' in prompt
            return json.dumps(answer)

    row = dict(item("a"), label="Composition")
    target, evidence = grounded_target(Teacher(), row, "proposer")
    assert (target["quote_a"], target["quote_b"]) == ("First a", "Second a")
    assert evidence["excerpt_selection"] == dict(quote_a_id="A1", quote_b_id="B1")
    assert len(evidence["invalid_generation_attempts"]) == 2


def test_chronological_entrypoint_refreshes_from_parent_and_resumes_final(
    tmp_path, monkeypatch
):
    from haireder.experiments import runner
    from haireder.utils.io import write_json

    def cohort(i, split, n=1):
        result = []
        for j in range(n):
            r = item(f"{i}_{j}", window=f"W{i}")
            result.append(
                dict(
                    r,
                    split=split,
                    label=("Workflow" if j % 2 else "Composition"),
                    fixture_index=j,
                )
            )
        return result

    splits = dict(
        train=cohort(1, "train", 6),
        validation=cohort(4, "validation", 24),
        online=[r for i in range(5, 29) for r in cohort(i, "online")],
        test=[r for i in range(29, 33) for r in cohort(i, "test", 2)],
    )
    monkeypatch.setattr(
        runner,
        "dataset_files",
        lambda path: (
            {"artifacts": {"test": "fixture"}, "negative_mining_backend": "random"},
            splits,
        ),
    )
    monkeypatch.setattr(runner, "implementation_hash", lambda: "integration_fixture")
    monkeypatch.setattr(runner, "ExampleIndex", lambda records, *args: list(records))
    trained = []

    def fit(rows, d, model, cfg, *, parents, single):
        assert len(rows) == len({r["pair_id"] for r in rows})
        trained.append((Path(d), parents))
        for role in ("proposer", "critic", "judge"):
            write_json(d / role / "metadata.json", dict(consumed=len(rows)))

    monkeypatch.setattr(runner, "train_roles", fit)

    class Predictor:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, r, index):
            assert all(int(p["window"][1:]) < int(r["window"][1:]) for p in index or [])
            out = {
                k: r[k]
                for k in ("pair_id", "issue_i", "issue_j", "repo", "split", "window")
            }
            assert "label" not in r and "gold_label" not in r
            c = 0.9 if int(r["pair_id"].split("_")[-1]) % 2 == 0 else 0.1
            return dict(
                out,
                predicted_label="Composition",
                s1=c,
                s2=c,
                s3=c,
                confidence=c,
                transcript={
                    "judge": {"parsed_output": {"justification": "test fixture"}}
                },
            )

    monkeypatch.setattr(runner, "RoleSystem", Predictor)
    plan = plans("no_augmentation", 0.0001, 0.0002, 4)[0]
    cfg = load_yaml("configs/experiments/formal_full_pipeline.yaml")
    directory = tmp_path / "control"
    runner.execute(
        plan,
        directory,
        tmp_path,
        cfg,
        {"device_preference": ["cpu"]},
        with_baselines=False,
    )
    assert len(trained) == 7 and trained[0][1] is None
    assert all(
        parents["proposer"] == trained[i - 1][0] / "proposer"
        for i, (_, parents) in enumerate(trained)
        if i
    )
    summary = json.loads((directory / "summary.json").read_text())
    assert (
        summary["held_out"]["count"] == 8
        and summary["workload"]["total_review_count"] == 0
    )
    assert (directory / "W9/refresh_comparison.json").exists() and (
        directory / "w29_before_refresh.jsonl"
    ).exists()
    runner.execute(
        plan,
        directory,
        tmp_path,
        cfg,
        {"device_preference": ["cpu"]},
        with_baselines=False,
        from_stage="final_evaluation",
    )
    assert len(trained) == 7


def test_creation_text_reverses_complete_history_and_excludes_incomplete():
    path = ROOT / "scripts/data"
    sys.path.insert(0, str(path))
    try:
        import build_creation_text
    finally:
        sys.path.pop(0)
    fields = dict(
        summary="edited",
        description="later",
        created="2018-01-11T00:00:00Z",
        updated="2018-02-10T00:00:00Z",
    )
    history = {
        "created": "2018-02-10T00:00:00Z",
        "id": "1",
        "items": [
            {"field": "summary", "fromString": "initial"},
            {"field": "description", "fromString": None},
        ],
    }
    issue = dict(fields=fields, changelog=dict(startAt=0, total=1, histories=[history]))
    assert build_creation_text.creation_text(issue) == dict(
        summary="initial", description=""
    )
    issue["changelog"]["total"] = 2
    assert build_creation_text.creation_text(issue) is None


def test_calendar_exclusive_end_and_empty_sprints():
    from haireder.data.windowing import build_fixed_sprints

    row = lambda i, t: dict(pair_id=i, pair_available_at=t)
    rows, windows = build_fixed_sprints(
        [row("end", "2018-02-22T00:00:00Z"), row("second", "2018-01-25T00:00:00Z")],
        "2018-01-11",
        "2018-02-22",
        14,
        {"window_count": 3, "train_windows": ["W1", "W2", "W3"]},
    )
    assert [w.count for w in windows] == [0, 1, 0] and [r["pair_id"] for r in rows] == [
        "second"
    ]


def test_text_budget_keeps_both_issue_ends_and_complete_debate():
    from haireder.training.context import fit_prompt
    from haireder.agents import render_judge_prompt

    class Tok:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    row = item("a")
    row["issue_i"]["text"] = (
        "A-START {{issue_j_text}}\n```python\ncode\n```\n" + "xyz" * 1500 + " A-END"
    )
    row["issue_j"]["text"] = "B-START " + "qst" * 1500 + " B-END"
    template = (ROOT / "prompts/agents/judge_prompt.md").read_text()
    prompt = render_judge_prompt(
        row, "entire debate with both reasoned arguments", template
    )
    from haireder.training.context import _SECTIONS

    assert [m["body"] for m in _SECTIONS.finditer(prompt)] == [
        row[k]["text"] for k in ("issue_i", "issue_j")
    ]
    fitted = fit_prompt(prompt, Tok(), 1500)
    assert len(fitted) <= 1500 and all(
        t in fitted
        for t in (
            "A-START",
            "A-END",
            "B-START",
            "B-END",
            "entire debate with both reasoned arguments",
        )
    )


def test_transcript_arguments_survive_judge_serialization():
    from haireder.agents.judge import compact_transcript_for_judge

    r = item("a")
    r.update(
        rounds=[
            {
                "round_index": 0,
                "proposer": {
                    "parsed_output": dict(label="None", rationale="left evidence")
                },
                "critic": {
                    "parsed_output": dict(label="Workflow", rationale="right evidence")
                },
            }
        ],
        termination_reason="round_budget",
    )
    compact = compact_transcript_for_judge(r)
    assert "left evidence" in compact and "right evidence" in compact


def test_augmentation_rejects_disagreement_and_string_booleans():
    from haireder.augmentation.llm_validator import LLMConsistencyValidator
    from haireder.augmentation.pair_aware import _validate_llm_augmented_pair

    class Independent:
        def generate(self, *args, **kwargs):
            return '{"is_valid":true,"label_matches":false}'

    validator = LLMConsistencyValidator(Independent(), "Check")

    class Filter:
        def filter(self, *args):
            return {"is_valid": True}

    seed = dict(item("a"), label="Workflow")
    decision = _validate_llm_augmented_pair(seed, seed, "Workflow", validator, Filter())
    assert not decision["is_valid"] and not decision["label_matches"]
    parsed = validator._parse_llm_output('{"is_valid":"false","label_matches":"true"}')
    assert not parsed["is_valid"] and not parsed["label_matches"]


def test_pair_tfidf_preserves_order_and_two_feature_spaces():
    import numpy as np
    from haireder.baselines.tfidf_logreg import OrderedPairTfidf

    v = OrderedPairTfidf()
    matrix = v.fit_transform(
        [("left word", "right term"), ("right term", "left word")]
    ).toarray()
    n = len(v.vocabulary_)
    assert (
        matrix.shape == (2, 2 * n)
        and np.allclose(matrix[0, :n], matrix[1, n:])
        and not np.allclose(matrix[0], matrix[1])
    )


def test_verified_supervision_overrides_export_gold_and_rejects_raw_online():
    from haireder.experiments.baselines import cumulative_supervision

    t = dict(item("train"), split="train", label="Workflow")
    o = dict(
        item("online"),
        split="online",
        label="None",
        accepted_label="Composition",
        decision_path="low_review",
        human_reviewed=True,
        review_detail=dict(
            pair_id="online",
            accepted_label="Composition",
            reviewer="test_fixture",
            reviewed_at="2026-11-22T12:00:00Z",
            rationale="fixture evidence",
            item_hash="test_hash",
            batch_id="test_batch",
        ),
    )
    test = dict(item("test"), split="test", label="Workflow")
    assert cumulative_supervision([t], [o], [test])[-1]["label"] == "Composition"
    with pytest.raises(ValueError):
        cumulative_supervision([t], [dict(o, analyst_simulation=True)], [test])
    with pytest.raises(ValueError):
        cumulative_supervision([t], [item("raw")], [test])
    with pytest.raises(ValueError, match="signed"):
        cumulative_supervision([t], [dict(o, review_detail={})], [test])


def test_baseline_rejects_nonfinite_scores_and_marks_diagnostic_coverage(tmp_path):
    from haireder.experiments.baselines import prediction, run_baseline

    target = dict(item("test"), split="test", label="None")
    with pytest.raises(ValueError, match="Invalid class probabilities"):
        prediction(target, [0.7, float("nan"), 0.1, 0.1, 0.04, 0.06], "broken")
    training = [
        dict(item("train" + str(i)), split="train", label=label)
        for i, label in enumerate(("None", "Workflow"))
    ]
    run_baseline("tfidf", training, [], [target], tmp_path, test_total=4)
    record = json.loads((tmp_path / "supervision.json").read_text())
    assert (record["test_scope"], record["evaluated"], record["test_total"]) == (
        "diagnostic_subset",
        1,
        4,
    )


def test_formal_role_training_rejects_zero_step_output(tmp_path, monkeypatch):
    from haireder.experiments import roles

    monkeypatch.setattr(roles, "create_llm_client", lambda **kwargs: object())
    monkeypatch.setattr(
        roles, "grounded_target", lambda *args: ({"label": "Workflow"}, {})
    )
    monkeypatch.setattr(
        roles, "train_role_adapter", lambda *args, **kwargs: {"optimizer_steps": 0}
    )
    rows = [dict(item("training"), split="train", label="Workflow")]
    with pytest.raises(ValueError, match="No verified trained adapter for proposer"):
        roles.train_roles(
            rows,
            tmp_path,
            {"training": {}},
            load_yaml("configs/experiments/formal_full_pipeline.yaml"),
            single=True,
        )
    assert not (tmp_path / "proposer_training.json").exists()


def test_replay_contains_all_new_and_distinct_history():
    from haireder.online_learning.refresh_corpus import build_refresh_corpus

    new = [dict(item(str(i)), accepted_label="Composition") for i in range(7)]
    past = [
        dict(item(str(i)), accepted_label="Workflow", transcript={"rounds": [1]})
        for i in range(14)
    ]
    rows, detail = build_refresh_corpus(new, past, augment_to_balance=False)
    assert (detail["selected_new_rows"], detail["selected_replay_rows"]) == (7, 3)
    assert len({r["pair_id"] for r in rows}) == 10 and all(
        r["transcript"]["rounds"] for r in rows[7:]
    )
