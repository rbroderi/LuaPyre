"""Enforce the three-program performance/admission gate on two A/B reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from speed_040_ab import GENERALITY_GROUPS


def evaluate(
    baseline: dict, candidate: dict, *, minimum_improvement: float = 0.02
) -> dict:
    groups = {}
    for path, names in GENERALITY_GROUPS.items():
        cases = {}
        for name in names:
            old = baseline["workloads"][name]["steady"]["luapyre"]["median_ms"]
            new = candidate["workloads"][name]["steady"]["luapyre"]["median_ms"]
            admissions = candidate["workloads"][name].get(
                "fast_path_admissions", {}
            ).get(path, 0)
            cases[name] = {
                "baseline_ms": old,
                "candidate_ms": new,
                "improvement_fraction": old / new - 1.0,
                "admissions": admissions,
                "passes": admissions > 0 and old / new - 1.0 >= minimum_improvement,
            }
        passed = sum(case["passes"] for case in cases.values())
        groups[path] = {
            "cases": cases,
            "passing_cases": passed,
            "status": "accepted" if passed >= 3 else "experimental",
        }
    return {
        "schema_version": 1,
        "minimum_improvement_fraction": minimum_improvement,
        "groups": groups,
        "accepted": all(group["status"] == "accepted" for group in groups.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.02,
        help="minimum improvement required in each case (default: 0.02)",
    )
    args = parser.parse_args()
    if args.minimum_improvement <= 0:
        parser.error("--minimum-improvement must be positive")
    report = evaluate(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
        minimum_improvement=args.minimum_improvement,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json is not None:
        args.json.write_text(rendered)
    print(rendered, end="")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
