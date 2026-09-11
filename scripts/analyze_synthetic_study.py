#!/usr/bin/env python3
"""Aggregate the frozen synthetic-study runs without modifying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

EXPECTED_FAMILIES = {
    "replications": 18,
    "misspecification": 12,
    "costs": 6,
    "critical-ablations": 12,
    "base": 1,
    "negative-control-shuffled-calibration": 1,
    "ablation-no-dose-penalty": 1,
    "ablation-no-time-penalty": 1,
    "ablation-no-registration-gate": 1,
}

METRICS = {
    "false_support_conditional": "lower",
    "coverage": "higher",
    "selective_verdict_risk": "lower",
    "supported_prediction_risk": "lower",
    "brier": "lower",
    "ece10": "lower",
    "correct_verdict_fraction": "higher",
    "mean_dose": "lower",
    "mean_acquisition_seconds": "lower",
}

CONFIG_FIELDS = [
    "seed",
    "source",
    "model",
    "target",
    "train_n",
    "calibration_n",
    "development_n",
    "test_n",
    "image_size",
    "train_correlation",
    "calibration_correlation",
    "development_correlation",
    "test_correlation",
    "nuisance_strength",
    "test_temporal_change",
    "test_unknown_nuisance",
    "n_states",
    "dose_weight",
    "time_weight",
    "support_threshold",
    "falsify_threshold",
    "shuffle_calibration_targets",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    fields = []
    seen = set()

    preferred = [
        "family",
        "run_id",
        "budget_steps",
        "policy",
        "baseline",
        "metric",
        "seed",
        "model",
        "nuisance_strength",
    ]

    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def config_metadata(config: dict) -> dict:
    row = {key: config.get(key) for key in CONFIG_FIELDS}

    gates = config.get("gates", {})
    row["registration_gate"] = gates.get("registration")
    row["initial_ood_gate"] = gates.get("initial_ood")
    row["min_history_support"] = gates.get("min_history_support")

    row["budgets"] = json.dumps(config.get("budgets", []))
    row["policies"] = json.dumps(config.get("policies", []))

    return row


def verify_run(run: Path) -> list[str]:
    receipt_path = run / "artifact_hashes.json"

    if not receipt_path.is_file():
        return ["artifact_hashes.json"]

    receipts = json.loads(receipt_path.read_text())
    damaged = []

    for relative, expected in receipts.items():
        path = run / relative
        if not path.is_file() or digest(path) != expected:
            damaged.append(relative)

    return damaged


def classify_family(run: Path, root: Path) -> str:
    relative = run.relative_to(root)

    if not relative.parts:
        raise ValueError(f"Cannot classify {run}")

    return relative.parts[0]


def outcome(delta: float | None, direction: str) -> str:
    if delta is None:
        return "undefined"

    if abs(delta) < 1e-12:
        return "tie"

    if direction == "lower":
        return "eig_better" if delta < 0 else "eig_worse"

    return "eig_better" if delta > 0 else "eig_worse"


def ci_interpretation(low, high) -> str:
    if low is None or high is None:
        return "unavailable"

    if high < 0:
        return "eig_better_ci95"

    if low > 0:
        return "eig_worse_ci95"

    return "inconclusive_ci95"


def summarize_delta_groups(
    rows: list[dict],
    group_fields: list[str],
) -> list[dict]:
    grouped = defaultdict(list)

    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)

    output = []

    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        deltas = [
            float(row["delta_eig_minus_baseline"])
            for row in values
            if row["delta_eig_minus_baseline"] is not None
        ]

        record = dict(zip(group_fields, key))
        record["n_configurations"] = len(values)
        record["n_defined"] = len(deltas)

        if deltas:
            record["mean_delta"] = statistics.mean(deltas)
            record["median_delta"] = statistics.median(deltas)
            record["min_delta"] = min(deltas)
            record["max_delta"] = max(deltas)
            record["sd_delta"] = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        else:
            record["mean_delta"] = None
            record["median_delta"] = None
            record["min_delta"] = None
            record["max_delta"] = None
            record["sd_delta"] = None

        record["eig_better"] = sum(row["directional_result"] == "eig_better" for row in values)
        record["tie"] = sum(row["directional_result"] == "tie" for row in values)
        record["eig_worse"] = sum(row["directional_result"] == "eig_worse" for row in values)

        defined_direction = record["eig_better"] + record["tie"] + record["eig_worse"]
        record["eig_better_fraction"] = (
            record["eig_better"] / defined_direction if defined_direction else None
        )

        record["ci95_eig_better"] = sum(
            row.get("ci95_result") == "eig_better_ci95" for row in values
        )
        record["ci95_eig_worse"] = sum(row.get("ci95_result") == "eig_worse_ci95" for row in values)
        record["ci95_inconclusive"] = sum(
            row.get("ci95_result") == "inconclusive_ci95" for row in values
        )

        output.append(record)

    return output


def plot_delta_summary(rows: list[dict], family: str, metric: str, out: Path) -> bool:
    selected = [
        row
        for row in rows
        if row["family"] == family and row["metric"] == metric and row["mean_delta"] is not None
    ]

    if not selected:
        return False

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    baselines = sorted({row["baseline"] for row in selected})
    budgets = sorted({int(row["budget_steps"]) for row in selected})

    fig, ax = plt.subplots(figsize=(9, 5))

    for baseline in baselines:
        subset = {int(row["budget_steps"]): row for row in selected if row["baseline"] == baseline}

        xs = [budget for budget in budgets if budget in subset]
        ys = [subset[budget]["mean_delta"] for budget in xs]

        if xs:
            ax.plot(xs, ys, marker="o", label=baseline)

    ax.axhline(0, linewidth=1)

    direction = METRICS[metric]
    favor = "< 0 favors EIG" if direction == "lower" else "> 0 favors EIG"

    ax.set_xlabel("Acquisition budget")
    ax.set_ylabel(f"EIG minus baseline ({favor})")
    ax.set_title(f"{family}: {metric}")
    ax.set_xticks(budgets)
    ax.legend(fontsize=8)
    fig.tight_layout()

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{family}_{metric}.png", dpi=180)
    plt.close(fig)

    return True


def markdown_table(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]

    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]

    for row in rows:
        values = []

        for field in fields:
            value = row.get(field)

            if isinstance(value, float):
                values.append(f"{value:.5g}")
            elif value is None:
                values.append("undefined")
            else:
                values.append(str(value))

        output.append("| " + " | ".join(values) + " |")

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    status_paths = sorted(root.rglob("status.json"))
    inventory = []
    result_rows = []
    stored_pairwise = []
    delta_rows = []
    problems = []

    family_counts = defaultdict(int)
    source_hashes = set()
    total_episodes = 0

    for status_path in status_paths:
        run = status_path.parent
        status = json.loads(status_path.read_text())

        if status.get("state") != "COMPLETE":
            continue

        required = [
            "config.lock.json",
            "summary.json",
            "paired_comparisons.json",
            "artifact_hashes.json",
        ]

        missing = [name for name in required if not (run / name).is_file()]

        if missing:
            problems.append(
                {
                    "run": str(run),
                    "problem": "missing files",
                    "details": ", ".join(missing),
                }
            )
            continue

        damaged = verify_run(run)

        if damaged:
            problems.append(
                {
                    "run": str(run),
                    "problem": "artifact checksum mismatch",
                    "details": ", ".join(damaged),
                }
            )
            continue

        family = classify_family(run, root)
        family_counts[family] += 1

        lock = json.loads((run / "config.lock.json").read_text())
        config = lock["config"]
        source_sha = lock.get("source_sha256")

        if source_sha:
            source_hashes.add(source_sha)

        run_id = str(run.relative_to(root))
        episodes = int(status.get("episodes", 0))
        total_episodes += episodes

        meta = {
            "family": family,
            "run_id": run_id,
            "run_path": str(run),
            "episodes": episodes,
            "source_sha256": source_sha,
            **config_metadata(config),
        }

        inventory.append(
            {
                **meta,
                "state": status["state"],
                "artifacts_valid": True,
            }
        )

        summaries = json.loads((run / "summary.json").read_text())
        comparisons = json.loads((run / "paired_comparisons.json").read_text())

        for summary_row in summaries:
            result_rows.append({**meta, **summary_row})

        for comparison in comparisons:
            ci = comparison.get("ci95", [None, None])

            stored_pairwise.append(
                {
                    **meta,
                    **comparison,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                }
            )

        summary_map = {(row["policy"], int(row["budget_steps"])): row for row in summaries}

        comparison_map = {
            (int(row["budget_steps"]), row["second"]): row
            for row in comparisons
            if row["first"] == "eig"
        }

        budgets = sorted({budget for policy, budget in summary_map if policy == "eig"})

        policies = sorted({policy for policy, _budget in summary_map if policy != "eig"})

        for budget in budgets:
            eig = summary_map.get(("eig", budget))

            if eig is None:
                continue

            for baseline in policies:
                other = summary_map.get((baseline, budget))

                if other is None:
                    continue

                bootstrap = comparison_map.get((budget, baseline))
                ci = bootstrap.get("ci95", [None, None]) if bootstrap else [None, None]

                for metric, direction in METRICS.items():
                    eig_value = eig.get(metric)
                    baseline_value = other.get(metric)

                    if eig_value is None or baseline_value is None:
                        delta = None
                    else:
                        delta = float(eig_value) - float(baseline_value)

                    record = {
                        **meta,
                        "budget_steps": budget,
                        "baseline": baseline,
                        "metric": metric,
                        "direction": direction,
                        "eig_value": eig_value,
                        "baseline_value": baseline_value,
                        "delta_eig_minus_baseline": delta,
                        "directional_result": outcome(delta, direction),
                        "ci95_low": None,
                        "ci95_high": None,
                        "ci95_result": "not_computed",
                    }

                    if metric == "false_support_conditional" and bootstrap:
                        record["ci95_low"] = ci[0]
                        record["ci95_high"] = ci[1]
                        record["ci95_result"] = ci_interpretation(ci[0], ci[1])

                    delta_rows.append(record)

    expected_errors = []

    if args.strict:
        for family, expected in EXPECTED_FAMILIES.items():
            observed = family_counts.get(family, 0)

            if observed != expected:
                expected_errors.append(f"{family}: expected {expected}, observed {observed}")

    if problems or expected_errors:
        write_csv(out / "problems.csv", problems)

        print("ANALYSIS REFUSED")
        for problem in problems:
            print(problem)
        for problem in expected_errors:
            print(problem)

        return 1

    family_summary = summarize_delta_groups(
        delta_rows,
        ["family", "budget_steps", "baseline", "metric"],
    )

    replication_stratified = summarize_delta_groups(
        [row for row in delta_rows if row["family"] == "replications"],
        [
            "model",
            "nuisance_strength",
            "budget_steps",
            "baseline",
            "metric",
        ],
    )

    misspecification_stratified = summarize_delta_groups(
        [row for row in delta_rows if row["family"] == "misspecification"],
        [
            "test_correlation",
            "test_temporal_change",
            "test_unknown_nuisance",
            "budget_steps",
            "baseline",
            "metric",
        ],
    )

    cost_stratified = summarize_delta_groups(
        [row for row in delta_rows if row["family"] == "costs"],
        [
            "dose_weight",
            "time_weight",
            "budget_steps",
            "baseline",
            "metric",
        ],
    )

    write_csv(out / "run_inventory.csv", inventory)
    write_csv(out / "master_results.csv", result_rows)
    write_csv(out / "stored_pairwise_bootstrap.csv", stored_pairwise)
    write_csv(out / "eig_vs_baselines.csv", delta_rows)
    write_csv(out / "family_summary.csv", family_summary)
    write_csv(out / "replication_stratified.csv", replication_stratified)
    write_csv(out / "misspecification_stratified.csv", misspecification_stratified)
    write_csv(out / "cost_stratified.csv", cost_stratified)

    figure_count = 0

    for family in ["replications", "misspecification", "costs"]:
        for metric in [
            "false_support_conditional",
            "coverage",
            "selective_verdict_risk",
        ]:
            if plot_delta_summary(
                family_summary,
                family,
                metric,
                out,
            ):
                figure_count += 1

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None

    manifest = {
        "root": str(root),
        "run_count": len(inventory),
        "total_episodes": total_episodes,
        "family_counts": dict(sorted(family_counts.items())),
        "source_hashes": sorted(source_hashes),
        "analysis_git_sha": git_sha,
        "figure_count": figure_count,
        "problems": problems,
        "strict_expected": EXPECTED_FAMILIES if args.strict else None,
    }

    (out / "study_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = [
        "# Master synthetic-study analysis",
        "",
        "**This is a computational study, not biological or hardware validation.**",
        "",
        f"- Complete runs analyzed: **{len(inventory)}**",
        f"- Decision episodes represented: **{total_episodes:,}**",
        f"- Distinct research-source hashes: **{len(source_hashes)}**",
        f"- Generated master figures: **{figure_count}**",
        "",
        "## Family inventory",
        "",
    ]

    inventory_table = [
        {
            "family": family,
            "runs": count,
        }
        for family, count in sorted(family_counts.items())
    ]

    report += markdown_table(inventory_table, ["family", "runs"])

    for family in ["replications", "misspecification", "costs"]:
        report += [
            "",
            f"## {family.title()} — budget 3",
            "",
            "### False-support difference",
            "",
            "Negative `mean_delta` favors EIG.",
            "",
        ]

        fs_rows = [
            row
            for row in family_summary
            if row["family"] == family
            and int(row["budget_steps"]) == 3
            and row["metric"] == "false_support_conditional"
        ]

        report += markdown_table(
            fs_rows,
            [
                "baseline",
                "n_configurations",
                "mean_delta",
                "median_delta",
                "eig_better",
                "eig_worse",
                "ci95_eig_better",
                "ci95_eig_worse",
                "ci95_inconclusive",
            ],
        )

        report += [
            "",
            "### Coverage difference",
            "",
            "Positive `mean_delta` favors EIG.",
            "",
        ]

        coverage_rows = [
            row
            for row in family_summary
            if row["family"] == family
            and int(row["budget_steps"]) == 3
            and row["metric"] == "coverage"
        ]

        report += markdown_table(
            coverage_rows,
            [
                "baseline",
                "n_configurations",
                "mean_delta",
                "median_delta",
                "eig_better",
                "eig_worse",
            ],
        )

    report += [
        "",
        "## Interpretation rules",
        "",
        "- Do not call EIG superior from false-support alone; inspect coverage and selective risk.",
        "- For false support, a paired CI entirely below zero favors EIG within that run.",
        "- Grid cells are experimental configurations, not independent biological specimens.",
        "- The cross-configuration means in this report are descriptive, not a pooled clinical CI.",
        "- Failed or undefined denominators must remain visible; do not silently replace them by zero.",
        "- The shuffled-calibration run is a negative control, not part of the replication estimate.",
        "- Do not tune a final claim by selecting only favorable configurations.",
        "",
        "## Next scientific gate",
        "",
        "Use these tables to decide whether the synthetic result is robust enough to justify",
        "the BBBC006 real-image replay. Freeze the synthetic conclusion before inspecting",
        "the final real-image test split.",
        "",
    ]

    (out / "REPORT.md").write_text("\n".join(report))

    print(json.dumps(manifest, indent=2))
    print()
    print("MASTER ANALYSIS COMPLETE")
    print(f"Report: {out / 'REPORT.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
