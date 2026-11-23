"""Prepare a blinded intake for a real chronological window before its confidence routing is available."""

import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from haireder.experiments.review import fingerprint, review_queue
from haireder.utils.io import read_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--online", default="data/processed/splits/formal_online.jsonl", type=Path
    )
    p.add_argument("--window", default="W5")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    rows = [
        r
        for r in read_jsonl(args.online)
        if r["window"] == args.window and r["split"] == "online"
    ]
    if not rows:
        raise ValueError("No online rows in the requested window")
    shown = [
        {k: r[k] for k in ("pair_id", "repo", "window", "issue_i", "issue_j")}
        for r in rows
    ]
    path = review_queue(args.out, fingerprint(shown), "online_intake", shown)
    print(f"{len(rows)} actual pairs pending formal routing in {path}")
