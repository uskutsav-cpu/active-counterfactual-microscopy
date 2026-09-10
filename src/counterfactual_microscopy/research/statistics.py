"""Explicit metric denominators and specimen-paired group bootstrap."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .validation import integer


def ratio(a, b):
    return float(a / b) if b else None


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty results")
    y = np.array([r["target"] for r in rows])
    p = np.array([r["posterior"] for r in rows])
    v = np.array([r["verdict"] for r in rows])
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid targets/posteriors")
    if not np.isin(v, ["supported", "falsified", "abstain"]).all():
        raise ValueError("invalid verdict")
    s = v == "supported"
    f = v == "falsified"
    resolved = s | f
    wrong = s & (y == 0) | f & (y == 1)
    errors = int(np.sum(s & (y == 0)))
    dose = sum(r["dose"] for r in rows)
    sec = sum(r["acquisition_seconds"] for r in rows)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= 1)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {
        "n": len(rows),
        "n_groups": len({r["group_id"] for r in rows}),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "supported": int(s.sum()),
        "falsified": int(f.sum()),
        "abstain": int((~resolved).sum()),
        "coverage": float(resolved.mean()),
        "false_support_conditional": ratio(errors, np.sum(y == 0)),
        "false_support_joint": errors / len(y),
        "supported_prediction_risk": ratio(errors, s.sum()),
        "falsification_recall": ratio(np.sum(f & (y == 0)), np.sum(y == 0)),
        "false_falsification_conditional": ratio(np.sum(f & (y == 1)), np.sum(y == 1)),
        "selective_verdict_risk": ratio(wrong.sum(), resolved.sum()),
        "correct_verdict_fraction": float(np.sum(resolved & ~wrong) / len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece10": float(ece),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "mean_dose": dose / len(y),
        "mean_acquisition_seconds": sec / len(y),
        "resolved_per_dose_proxy": ratio(resolved.sum(), dose),
        "resolved_per_second": ratio(resolved.sum(), sec),
        "ineligible_fraction": float(np.mean([not r.get("eligible", True) for r in rows])),
    }


def risk_coverage(rows: list[dict]) -> list[dict]:
    p = np.array([r["posterior"] for r in rows])
    y = np.array([r["target"] for r in rows])
    eligible = np.array([r.get("eligible", True) for r in rows])
    confidence = np.maximum(p, 1 - p)
    out = []
    for t in sorted(set(confidence[eligible]), reverse=True):
        take = eligible & (confidence >= t)
        out.append(
            {
                "threshold": float(t),
                "coverage": float(take.mean()),
                "risk": float(np.mean((p[take] >= 0.5) != y[take])),
            }
        )
    return out


def summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["policy"], r["budget_steps"]].append(r)
    return [{"policy": p, "budget_steps": b, **metrics(v)} for (p, b), v in sorted(groups.items())]


def paired_bootstrap(
    first: list[dict], second: list[dict], metric: str, repeats: int = 500, seed: int = 42
) -> dict:
    integer("repeats", repeats, 1)
    a = {r["specimen_id"]: r for r in first}
    b = {r["specimen_id"]: r for r in second}
    if len(a) != len(first) or len(b) != len(second) or set(a) != set(b) or (not a):
        raise ValueError("need identical specimen sets, one row per policy/budget/specimen")
    grouped = defaultdict(list)
    for k in sorted(a):
        if a[k]["group_id"] != b[k]["group_id"] or a[k]["target"] != b[k]["target"]:
            raise ValueError("pair mismatch")
        grouped[a[k]["group_id"]].append(k)
    groups = list(grouped)
    if len(groups) < 2:
        raise ValueError("bootstrap requires at least two independent groups")

    def delta(keys):
        ma = metrics([a[k] for k in keys])[metric]
        mb = metrics([b[k] for k in keys])[metric]
        return None if ma is None or mb is None else ma - mb

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        keys = [k for g in rng.choice(groups, len(groups), replace=True) for k in grouped[g]]
        value = delta(keys)
        if value is not None:
            values.append(value)
    return {
        "metric": metric,
        "difference_first_minus_second": delta(sorted(a)),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist()
        if len(values) >= max(10, repeats // 2)
        else [None, None],
        "groups": len(groups),
        "valid_replicates": len(values),
        "requested_replicates": repeats,
        "method": "paired percentile group bootstrap",
        "multiplicity_adjusted": False,
    }
