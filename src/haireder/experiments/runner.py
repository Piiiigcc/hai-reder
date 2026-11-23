"""Execute the full paper method, its three ablations, and the specified 64 configurations."""

from collections import Counter
from itertools import product
from pathlib import Path
import json

from haireder.data.formal import file_hash
from haireder.online_learning.refresh_corpus import build_refresh_corpus
from haireder.taxonomy import LABELS
from haireder.utils.config import resolve_path
from haireder.utils.io import read_jsonl, write_json, write_jsonl
from .augmentation import generate_pairs
from .evaluation import (
    INFERENCE_FIELDS,
    fit_bundle,
    calibrate,
    evaluation,
    validate_predictions,
    review_workload,
)
from .review import fingerprint
from .roles import RoleSystem, ExampleIndex, train_roles
from .sprint import records, process_sprint
from .stages import Stages, run_lock


FORMAL_WINDOWS = {
    "train": range(1, 4),
    "validation": range(4, 5),
    "online": range(5, 29),
    "test": range(29, 33),
}


def plans(suite, low=0.4, high=0.8, period=4):
    if not 0 <= low < high <= 1 or period < 1:
        raise ValueError("Invalid thresholds or refresh period")

    def entry(variant, l, h, k):
        name = f"{variant}_low{l:g}_high{h:g}_k{k}"
        return dict(
            name=name,
            variant=variant,
            low=l,
            high=h,
            period=k,
            single=variant == "single",
            dynamic=variant != "static",
            augmentation=variant != "no_augmentation",
            negatives="random" if variant == "no_augmentation" else "sbert",
        )

    if suite == "grid":
        return [
            entry("full", l, h, k)
            for l, h, k in product(
                (0.1, 0.2, 0.3, 0.4), (0.6, 0.7, 0.8, 0.9), (2, 4, 6, 8)
            )
        ]
    variants = (
        ("full", "single", "static", "no_augmentation")
        if suite == "ablations"
        else (suite,)
    )
    if any(v not in {"full", "single", "static", "no_augmentation"} for v in variants):
        raise ValueError(suite)
    return [entry(v, low, high, period) for v in variants]


def implementation_hash():
    root = resolve_path(".")
    paths = sorted((root / "src/haireder").rglob("*.py")) + sorted(
        (root / "prompts").rglob("*.md")
    )
    paths.append(root / "scripts/workflow/run_formal_pipeline.py")
    return fingerprint({str(p.relative_to(root)): file_hash(p) for p in paths})


def dataset_files(directory):
    directory = Path(directory)
    m = json.loads((directory / "formal_manifest.json").read_text())
    required = {f"formal_{split}.jsonl" for split in FORMAL_WINDOWS}
    if m.get("mode") != "formal" or required - m.get("artifacts", {}).keys():
        raise ValueError("Formal manifest lacks verified split files")
    for name, h in m["artifacts"].items():
        if file_hash(directory / name) != h:
            raise ValueError(f"Dataset artifact changed: {name}")
    for path, h in m["inputs"].items():
        if file_hash(resolve_path(path)) != h:
            raise ValueError(f"Dataset input changed: {path}")
    splits = {}
    seen = set()
    for split, windows in FORMAL_WINDOWS.items():
        rows = read_jsonl(directory / f"formal_{split}.jsonl")
        if not rows or len(rows) != m["split_counts"][split]:
            raise ValueError(f"Unexpected row count in formal {split} split")
        allowed = {f"W{i}" for i in windows}
        for row in rows:
            key = row.get("pair_id")
            if (
                not isinstance(key, str)
                or not key
                or key in seen
                or row.get("split") != split
                or row.get("window") not in allowed
                or row.get("label") not in LABELS
                or row.get("repo") not in m["repositories"]
            ):
                raise ValueError(f"Invalid or duplicate formal {split} row: {key}")
            seen.add(key)
        splits[split] = rows
    return m, splits


def artifact_files(directory):
    directory = Path(directory)
    return [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and "trainer_state" not in p.relative_to(directory).parts
        and not (p.parent.name == "human_review" and p.name == "review.csv")
    ]


def score(predictor, rows, pool, cfg, bundle, plan, *, labels=False):
    if not rows:
        return []
    s = cfg["confidence"]["sbert"]
    index = ExampleIndex(pool, s["model_name"], s["k"]) if pool else None
    results = []
    for row in rows:
        item = predictor.infer({k: row[k] for k in INFERENCE_FIELDS}, index)
        if bundle:
            item = calibrate(item, bundle, plan["low"], plan["high"])
        results.append(item)
    validate_predictions(rows, results, labelled=False)
    if labels:
        gold = {r["pair_id"]: r["label"] for r in rows}
        return [dict(item, gold_label=gold[item["pair_id"]]) for item in results]
    return results


def execute(
    plan,
    directory,
    dataset_dir,
    config,
    model_config,
    *,
    from_stage=None,
    checkpoint=None,
    with_baselines=True,
    device="cpu",
):
    directory = Path(directory).resolve()
    with run_lock(directory):
        manifest, splits = dataset_files(dataset_dir)
        if manifest["negative_mining_backend"] != plan["negatives"]:
            raise ValueError("Dataset mining backend disagrees with experiment variant")
        identity = fingerprint(
            [
                plan,
                manifest["artifacts"],
                manifest.get("inputs", {}),
                manifest.get("data_version"),
                config,
                model_config,
                implementation_hash(),
                device,
            ]
        )
        stages = Stages(directory, identity, from_stage, checkpoint)
        write_json(
            directory / "configuration.json",
            dict(
                plan=plan,
                identity=identity,
                model=model_config,
                dataset=manifest["artifacts"],
                dataset_sources=manifest.get("inputs", {}),
                dataset_version=manifest.get("data_version"),
                config=config,
                device=device,
            ),
        )
        train = splits["train"]
        online_cfg = config["online_learning"]
        roles = ("proposer",) if plan["single"] else ("proposer", "critic", "judge")
        val = splits["validation"]

        def training_stage(name, rows, parents, previous, initial):
            base = directory / name
            if plan["augmentation"]:
                out = base / "augmentation"

                def aug():
                    generate_pairs(
                        rows, out, config, human_sample=100 if initial else 0
                    )
                    return artifact_files(out)

                previous = stages.run(
                    name + "_augmentation",
                    "training" if initial else "online_learning",
                    [previous, fingerprint(rows)],
                    aug,
                )
                if initial and not (out / "human_review/review.csv").is_file():
                    raise FileNotFoundError(
                        f"Missing bound augmentation review sheet in {out}"
                    )
                rows = rows + read_jsonl(out / "qualified.jsonl")

            def fit():
                from haireder.training.lora_trainer import _prepare_device

                if (
                    model_config.get("quantization", {}).get("enabled")
                    and _prepare_device(model_config) != "cuda"
                ):
                    raise RuntimeError(
                        "This model configuration requires CUDA NF4 QLoRA"
                    )
                train_roles(
                    rows,
                    base / "adapters",
                    model_config,
                    config,
                    parents=parents,
                    single=plan["single"],
                )
                return artifact_files(base / "adapters")

            commit = stages.run(
                name,
                "training" if initial else "online_learning",
                [previous, fingerprint(rows)],
                fit,
            )
            return {r: base / "adapters" / r for r in roles}, commit

        # Seed calibration/retrieval exclusively with earlier labelled training windows.
        pool = records(
            [
                dict(
                    r,
                    accepted_label=r["label"],
                    review_source="initial_training",
                    confidence=1.0,
                    confidence_region="high",
                    analyst_status="initial",
                )
                for r in train
            ],
            0,
        )
        prev_counts = Counter(r["label"] for r in train)
        from haireder.online_learning.pool_manager import update_pool_snapshot

        pool, _ = update_pool_snapshot(
            [],
            pool,
            capacity=online_cfg["pool_capacity"],
            recency_partition_ratio=online_cfg["recency_partition_ratio"],
            difficulty_partition_ratio=online_cfg["difficulty_partition_ratio"],
            quota_label_counts=prev_counts,
        )
        active, commit = training_stage("initial", train, None, identity, True)

        def predictor(adapters):
            return RoleSystem(adapters, model_config, config, single=plan["single"])

        bundle = None
        if plan["dynamic"]:

            def calibration():
                rows = score(
                    predictor(active), val, pool, config, None, plan, labels=True
                )
                validate_predictions(val, rows)
                write_jsonl(directory / "w4_predictions.jsonl", rows)
                search = config["confidence"]["weight_search"]
                write_json(
                    directory / "calibration.json",
                    fit_bundle(rows, step=search["step"], n_bins=search["n_bins"]),
                )
                return [
                    directory / "w4_predictions.jsonl",
                    directory / "calibration.json",
                ]

            commit = stages.run("calibration", "calibration", commit, calibration)
            bundle = json.loads((directory / "calibration.json").read_text())
        accepted_all = []
        new = []
        history = pool
        pending_comparison = None
        for i in range(5, 29):
            w = f"W{i}"
            group = [r for r in splits["online"] if r["window"] == w]
            dest = directory / w

            def initial_predictions():
                raw = (
                    score(
                        predictor(active),
                        group,
                        pool if plan["dynamic"] else [],
                        config,
                        bundle,
                        plan,
                    )
                    if group
                    else []
                )
                write_jsonl(dest / "predictions.jsonl", raw)
                return [dest / "predictions.jsonl"]

            commit = stages.run(
                w + "_inference", "online_learning", commit, initial_predictions
            )
            raw = read_jsonl(dest / "predictions.jsonl")
            held_labels = {r["pair_id"]: r["label"] for r in group}
            actual = [dict(r, gold_label=held_labels[r["pair_id"]]) for r in raw]
            if group:
                validate_predictions(group, actual)
            commit = stages.run(
                w + "_metrics",
                "online_learning",
                commit,
                lambda: [write_json(dest / "metrics.json", evaluation(actual))],
            )
            if pending_comparison and group:

                def compare():
                    earlier = score(
                        predictor(pending_comparison),
                        group,
                        pool,
                        config,
                        bundle,
                        plan,
                        labels=True,
                    )
                    validate_predictions(group, earlier)
                    write_jsonl(dest / "before_refresh.jsonl", earlier)
                    write_json(
                        dest / "refresh_comparison.json",
                        dict(
                            window=w,
                            cohort=fingerprint(group),
                            before=evaluation(earlier),
                            after=evaluation(actual),
                        ),
                    )
                    return [
                        dest / "before_refresh.jsonl",
                        dest / "refresh_comparison.json",
                    ]

                commit = stages.run(
                    w + "_refresh_comparison", "online_learning", commit, compare
                )
                pending_comparison = None
            if not plan["dynamic"]:
                continue

            def finalize():
                def reinfer(rows, working):
                    return score(predictor(active), rows, working, config, bundle, plan)

                if raw:
                    process_sprint(
                        raw,
                        pool,
                        prev_counts,
                        online_cfg,
                        dest,
                        fingerprint([identity, w, commit]),
                        reinfer,
                        sequence=len(train) + len(accepted_all),
                    )
                else:
                    write_jsonl(dest / "accepted.jsonl", [])
                    write_jsonl(dest / "pool.jsonl", pool)
                    write_json(
                        dest / "summary.json",
                        dict(review_workload([]), accepted_counts={}),
                    )
                return [
                    dest / "accepted.jsonl",
                    dest / "pool.jsonl",
                    dest / "summary.json",
                    *sorted(dest.glob("*/review.csv")),
                    *sorted(dest.glob("*/items.jsonl")),
                    *sorted(dest.glob("*/batch.json")),
                    *sorted(dest.glob("reinference.json")),
                    *sorted(dest.glob("medium_predictions.jsonl")),
                ]

            commit = stages.run(w + "_decisions", "online_learning", commit, finalize)
            done = read_jsonl(dest / "accepted.jsonl")
            accepted_all.extend(done)
            new.extend(done)
            pool = read_jsonl(dest / "pool.jsonl")
            prev_counts = Counter(r["accepted_label"] for r in done)
            if (i - 4) % plan["period"] == 0 and new:
                corpus, summary = build_refresh_corpus(
                    new,
                    history,
                    augment_to_balance=False,
                    new_data_ratio=online_cfg["new_data_ratio"],
                    replay_ratio=online_cfg["replay_ratio"],
                    seed=42 + i,
                )
                selection = dict(
                    summary,
                    new_ids=[r["pair_id"] for r in new],
                    replay_ids=[r["pair_id"] for r in corpus[len(new) :]],
                )
                commit = stages.run(
                    w + "_refresh_sources",
                    "online_learning",
                    [commit, fingerprint(selection)],
                    lambda: [write_json(dest / "refresh_sources.json", selection)],
                )
                pending_comparison = active
                active, commit = training_stage(
                    w + "_refresh", corpus, active, commit, False
                )
                history = pool
                new = []
        test = splits["test"]

        def final():
            rows = score(
                predictor(active),
                test,
                pool if plan["dynamic"] else [],
                config,
                bundle,
                plan,
                labels=True,
            )
            validate_predictions(test, rows)
            paths = []
            write_jsonl(directory / "test_predictions.jsonl", rows)
            metrics = evaluation(rows)
            if pending_comparison:
                first = [r for r in test if r["window"] == "W29"]
                if first:
                    earlier = score(
                        predictor(pending_comparison),
                        first,
                        pool,
                        config,
                        bundle,
                        plan,
                        labels=True,
                    )
                    validate_predictions(first, earlier)
                    write_jsonl(directory / "w29_before_refresh.jsonl", earlier)
                    paths.append(directory / "w29_before_refresh.jsonl")
                    write_json(
                        directory / "w29_refresh_comparison.json",
                        dict(
                            window="W29",
                            cohort=fingerprint(first),
                            before=evaluation(earlier),
                            after=evaluation([r for r in rows if r["window"] == "W29"]),
                        ),
                    )
                    paths.append(directory / "w29_refresh_comparison.json")
            write_json(
                directory / "summary.json",
                dict(
                    held_out=metrics,
                    workload=review_workload(accepted_all),
                    cohort=fingerprint(test),
                    plan=plan,
                ),
            )
            return [
                directory / "test_predictions.jsonl",
                directory / "summary.json",
                *paths,
            ]

        commit = stages.run("final_evaluation", "final_evaluation", commit, final)
        if with_baselines and plan["variant"] == "full":
            from .baselines import run_baseline

            for key in ("tfidf", "bert", "zero_shot"):
                dest = directory / "baselines" / key

                def baseline():
                    run_baseline(
                        key,
                        train,
                        accepted_all,
                        test,
                        dest,
                        device=device,
                        model_cfg=model_config,
                    )
                    return artifact_files(dest)

                stages.run(key, "baselines", commit, baseline)
        if stages.seek:
            raise ValueError(f"Unknown start stage {stages.seek}")
        write_json(
            directory / "status.json",
            dict(
                status="complete",
                identity=identity,
                checkpoint=str(stages.path.resolve())
                if checkpoint
                else "checkpoint.json",
                checkpoint_sha256=file_hash(stages.path),
                configuration_sha256=file_hash(directory / "configuration.json"),
                completed_baselines=[
                    key
                    for key in ("tfidf", "bert", "zero_shot")
                    if key in stages.state["stages"]
                ],
            ),
        )
