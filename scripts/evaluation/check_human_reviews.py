"""Validate immutable batch binding and summarize actual review entries."""

import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from haireder.experiments.review import read_decisions, import_independent_decisions
from haireder.utils.io import read_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument(
        "--import-into-low",
        type=Path,
        help="Attach matching prior intake notes for a reviewer to consult before signing the low-confidence form",
    )
    args = p.parse_args()
    if args.import_into_low:
        print(
            f"Attached {import_independent_decisions(args.csv, args.import_into_low)} matching intake notes"
        )
    result = read_decisions(args.csv, require_all=False)
    manifest = json.loads((args.csv.parent / "batch.json").read_text())
    stats = dict(
        batch_id=manifest["batch_id"],
        kind=manifest["kind"],
        total=manifest["sample_size"],
        reviewed=len(result),
        pending=manifest["sample_size"] - len(result),
    )
    if manifest["kind"] == "augmentation":
        labels = {
            r["pair_id"]: r["label"]
            for r in read_jsonl(args.csv.parent.parent / "proposals.jsonl")
        }
        stats["preservation_rate"] = (
            sum(r["preserves_relation"].lower() == "yes" for r in result.values())
            / len(result)
            if result
            else None
        )
        stats["label_agreement"] = (
            sum(r["accepted_label"] == labels[k] for k, r in result.items())
            / len(result)
            if result
            else None
        )
        qualified = {
            r["pair_id"] for r in read_jsonl(args.csv.parent.parent / "qualified.jsonl")
        }
        sample = read_jsonl(args.csv.parent / "items.jsonl")
        groups = {}
        for name, passed in (("qualified", True), ("rejected", False)):
            keys = [
                r["pair_id"] for r in sample if (r["pair_id"] in qualified) == passed
            ]
            signed = [result[k] for k in keys if k in result]
            groups[name] = dict(
                sample=len(keys),
                reviewed=len(signed),
                preservation_rate=sum(
                    r["preserves_relation"].lower() == "yes" for r in signed
                )
                / len(signed)
                if signed
                else None,
            )
        stats["machine_filter_groups"] = groups
    print(json.dumps(stats, indent=2))
