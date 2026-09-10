"""Regenerate tables, figures, and a plain-language report from frozen result rows."""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np

from .io import atomic_json, write_csv
from .statistics import risk_coverage, summary


def render_report(directory: str | Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(directory)
    rows = json.loads((directory / "test_episodes.json").read_text())
    config = json.loads((directory / "config.lock.json").read_text())["config"]
    table = summary(rows)
    atomic_json(directory / "summary.json", table)
    write_csv(directory / "summary.csv", table)
    figure_dir = directory / "figures"
    figure_dir.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"], row["budget_steps"]].append(row)
    max_budget = max(row["budget_steps"] for row in rows)
    chosen = [r for r in table if r["budget_steps"] == max_budget]
    for metric, label in [
        ("false_support_conditional", "False support among negative target cases"),
        ("coverage", "Resolved-case fraction"),
        ("brier", "Brier score"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        vals = [np.nan if r[metric] is None else r[metric] for r in chosen]
        ax.bar([r["policy"] for r in chosen], vals)
        ax.set_ylabel(label)
        ax.set_title(f"{config['source']}: acquisition cap {max_budget}")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{metric}.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    curves = []
    for (policy, budget), values in sorted(grouped.items()):
        if budget != max_budget:
            continue
        curve = risk_coverage(values)
        curves.extend([{"policy": policy, "budget_steps": budget, **x} for x in curve])
        if curve:
            ax.plot([x["coverage"] for x in curve], [x["risk"] for x in curve], label=policy)
    ax.set(
        xlabel="Coverage (all test cases denominator)",
        ylabel="Risk of binary verifier verdict",
        title="Eligibility-gated risk–coverage; ties enter together",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if ax.lines:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "risk_coverage.png", dpi=140)
    plt.close(fig)
    atomic_json(directory / "risk_coverage.json", curves)
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in sorted({r["policy"] for r in table}):
        selected = sorted([r for r in table if r["policy"] == policy], key=lambda r: r["mean_dose"])
        ax.plot(
            [r["mean_dose"] for r in selected],
            [r["coverage"] for r in selected],
            marker="o",
            label=policy,
        )
    ax.set(
        xlabel="Mean dose proxy (not calibrated photons)",
        ylabel="Resolved-case fraction",
        title="Realized cost versus coverage",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "dose_coverage.png", dpi=140)
    plt.close(fig)
    label = (
        "Original prediction agrees with the reference task label"
        if config["target"] == "correctness"
        else "Independent binary hypothesis annotation (see provenance)"
    )
    header = [
        "policy",
        "budget_steps",
        "n",
        "coverage",
        "false_support_conditional",
        "supported_prediction_risk",
        "selective_verdict_risk",
        "brier",
        "mean_dose",
    ]

    def fmt(value):
        return (
            "undefined"
            if value is None
            else f"{value:.4g}"
            if isinstance(value, float)
            else str(value)
        )

    markdown = [
        "# Experiment report",
        "",
        "**Status: computational experiment, not biological or hardware validation.**",
        "",
        f"Verification target: {label}.",
        "",
        "Policies had the same action menu and hard caps, not necessarily identical realized dose/time.",
        "An all-abstain policy can have zero false-support errors; compare coverage and selective risk.",
        "The correctness target is not evidence that the model is biology-driven.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in table:
        markdown.append("| " + " | ".join(fmt(row[k]) for k in header) + " |")
    markdown += [
        "",
        "## Interpretation limits",
        "",
        "These baseline implementations are not claimed as reproductions of published methods.",
        "The joint evidence model assumes static, registered same-field panels.",
        "A supported verdict is conditional on the target, calibration distribution, action family, and quality gates.",
        "Undefined denominators are reported as null/undefined, never as zero.",
        "Bootstrap intervals are paired by group and not adjusted across policies/budgets.",
        "Do not select the strongest claim or the next configuration from a final held-out test.",
    ]
    (directory / "REPORT.md").write_text("\n".join(markdown) + "\n")
    html = '<html><meta charset="utf-8"><title>Counterfactual microscopy experiment</title><body style="font-family:system-ui;max-width:1100px;margin:40px auto;line-height:1.5">'
    html += "<h1>Counterfactual microscopy experiment</h1><p><strong>Computational results — not biological validation.</strong></p>"
    html += '<pre style="white-space:pre-wrap">' + escape("\n".join(markdown)) + "</pre>"
    for name in [
        "false_support_conditional",
        "coverage",
        "brier",
        "risk_coverage",
        "dose_coverage",
    ]:
        html += f'<img style="max-width:100%" src="figures/{name}.png" alt="{name}">'
    html += "</body></html>"
    (directory / "report.html").write_text(html)
    return {"summary_rows": len(table), "figures": 5, "report": str(directory / "REPORT.md")}
