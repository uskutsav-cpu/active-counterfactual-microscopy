"""Run calibrated 10-state Bayesian LSFM replay on development or test specimens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from counterfactual_microscopy.research.io import atomic_json, digest, environment, write_csv
from counterfactual_microscopy.research.optical_bayes import (
    ACTION_FEATURES,
    ACTION_OFFSETS_UM,
    BASE_FEATURES,
    POLICIES,
    STATE_OFFSETS_UM,
    MultiStateJoint,
    OpticalBayesModel,
    VectorQuantizer,
    choose_action,
    posterior_summary,
    split_specimens,
)


def stable_seed(*items: object) -> int:
    payload = json.dumps(items, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def read_response_matrix(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "specimen_id",
        "start_offset_um",
        "action_offset_um",
        "candidate_quality",
        *BASE_FEATURES,
        *ACTION_FEATURES,
    }
    if not rows:
        raise ValueError("empty response matrix")
    if not required.issubset(rows[0]):
        raise ValueError(f"response matrix missing fields: {sorted(required - set(rows[0]))}")
    for row in rows:
        start = int(float(row["start_offset_um"]))
        action = int(float(row["action_offset_um"]))
        if start not in STATE_OFFSETS_UM or action not in ACTION_OFFSETS_UM:
            raise ValueError("response matrix contains unexpected state/action")
    if len(rows) != 2520:
        raise ValueError(f"expected 2520 response rows; got {len(rows)}")
    return rows


def vector(row: dict, columns: tuple[str, ...]) -> np.ndarray:
    result = np.asarray([float(row[column]) for column in columns], dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("nonfinite feature vector")
    return result


def build_episode_index(rows: list[dict]) -> dict[tuple[str, int], dict[int, dict]]:
    episodes: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["specimen_id"], int(float(row["start_offset_um"])))
        action = int(float(row["action_offset_um"]))
        if action in episodes[key]:
            raise ValueError(f"duplicate episode/action: {key}, {action}")
        episodes[key][action] = row
    expected_actions = set(ACTION_OFFSETS_UM)
    for key, action_rows in episodes.items():
        if set(action_rows) != expected_actions:
            raise ValueError(f"incomplete episode {key}")
    if len(episodes) != 420:
        raise ValueError(f"expected 420 episodes; got {len(episodes)}")
    return dict(episodes)


def fit_model(
    episodes: dict[tuple[str, int], dict[int, dict]],
    calibration_specimens: tuple[str, ...],
    *,
    n_bins: int,
    seed: int,
    alpha: float,
) -> tuple[OpticalBayesModel, dict]:
    calibration_set = set(calibration_specimens)
    keys = sorted(key for key in episodes if key[0] in calibration_set)
    if len(keys) != 240:
        raise ValueError(f"expected 240 calibration episodes; got {len(keys)}")

    baseline_x = np.stack(
        [vector(next(iter(episodes[key].values())), BASE_FEATURES) for key in keys]
    )
    baseline_q = VectorQuantizer(n_bins, seed).fit(baseline_x)
    baseline_outcomes = baseline_q.transform(baseline_x)

    action_quantizers = []
    action_outcomes = []
    expected_quality = np.zeros((len(STATE_OFFSETS_UM), len(ACTION_OFFSETS_UM)), dtype=float)

    for action_index, action in enumerate(ACTION_OFFSETS_UM):
        x = np.stack([vector(episodes[key][action], ACTION_FEATURES) for key in keys])
        q = VectorQuantizer(n_bins, stable_seed(seed, "action", action)).fit(x)
        action_quantizers.append(q)
        action_outcomes.append(q.transform(x))

        for state_index, state in enumerate(STATE_OFFSETS_UM):
            qualities = [
                float(episodes[key][action]["candidate_quality"]) for key in keys if key[1] == state
            ]
            if len(qualities) != len(calibration_specimens):
                raise ValueError("candidate-quality calibration support mismatch")
            expected_quality[state_index, action_index] = float(np.mean(qualities))

    outcomes = np.column_stack([baseline_outcomes, *action_outcomes]).astype(int)
    targets = np.asarray([STATE_OFFSETS_UM.index(key[1]) for key in keys], dtype=int)
    cardinalities = (n_bins,) * (1 + len(ACTION_OFFSETS_UM))
    joint = MultiStateJoint(
        outcomes,
        targets,
        cardinalities,
        alpha=alpha,
        prior_alpha=1.0,
    )

    model = OpticalBayesModel(
        states_um=STATE_OFFSETS_UM,
        actions_um=ACTION_OFFSETS_UM,
        baseline_quantizer=baseline_q,
        action_quantizers=tuple(action_quantizers),
        joint=joint,
        expected_quality=expected_quality,
        calibration_specimens=calibration_specimens,
        n_bins=n_bins,
    )

    report = {
        "calibration_specimens": list(calibration_specimens),
        "calibration_episodes": len(keys),
        "n_bins": n_bins,
        "alpha": alpha,
        "baseline_bin_counts": np.bincount(baseline_outcomes, minlength=n_bins).tolist(),
        "action_bin_counts": {
            str(action): np.bincount(action_outcomes[i], minlength=n_bins).tolist()
            for i, action in enumerate(ACTION_OFFSETS_UM)
        },
        "state_counts": np.bincount(targets, minlength=len(STATE_OFFSETS_UM)).tolist(),
        "expected_quality": expected_quality.tolist(),
        "likelihood_model": "smoothed empirical full-joint discrete panel law",
        "conditional_independence_assumption": False,
    }
    return model, report


def transform_episode(
    model: OpticalBayesModel,
    action_rows: dict[int, dict],
) -> tuple[int, dict[int, int]]:
    first = action_rows[ACTION_OFFSETS_UM[0]]
    baseline_bin = model.baseline_outcome(vector(first, BASE_FEATURES))
    action_bins = {}
    for action_index, action in enumerate(ACTION_OFFSETS_UM):
        action_bins[action_index] = model.action_outcome(
            action_index,
            vector(action_rows[action], ACTION_FEATURES),
        )
    return baseline_bin, action_bins


def run_episode(
    model: OpticalBayesModel,
    specimen: str,
    true_state_um: int,
    action_rows: dict[int, dict],
    policy: str,
    budget: int,
    seed: int,
) -> dict:
    baseline_bin, action_bins = transform_episode(model, action_rows)
    history = {0: baseline_bin}
    used: set[int] = set()
    posterior = model.posterior(history)
    initial = posterior_summary(posterior, model.states_um, true_state_um)
    initial_support = model.joint.support(history)
    trace = []
    rng = np.random.default_rng(stable_seed(seed, specimen, true_state_um, policy, budget))

    for step in range(budget):
        action_index, scores = choose_action(
            policy,
            model,
            history,
            used,
            rng,
            true_state_um=true_state_um if policy == "oracle_focus" else None,
        )
        information_gain = model.information_gain(action_index, history)
        outcome = action_bins[action_index]
        posterior_before = posterior.copy()
        history[action_index + 1] = outcome
        used.add(action_index)
        posterior = model.posterior(history)
        trace.append(
            {
                "step": step + 1,
                "action_index": action_index,
                "action_offset_um": model.actions_um[action_index],
                "outcome_bin": outcome,
                "information_gain": information_gain,
                "posterior_before": posterior_before.tolist(),
                "posterior_after": posterior.tolist(),
                "calibration_support_after": model.joint.support(history),
                "candidate_scores": scores,
            }
        )

    final = posterior_summary(posterior, model.states_um, true_state_um)
    selected_offsets = [int(item["action_offset_um"]) for item in trace]
    initial_residual = abs(true_state_um)
    best_acquired_residual = (
        min(abs(true_state_um + action) for action in selected_offsets)
        if selected_offsets
        else initial_residual
    )
    last_acquired_residual = (
        abs(true_state_um + selected_offsets[-1]) if selected_offsets else initial_residual
    )
    available_oracle = min(abs(true_state_um + action) for action in model.actions_um)
    return {
        "specimen_id": specimen,
        "true_state_um": true_state_um,
        "policy": policy,
        "budget": budget,
        "baseline_bin": baseline_bin,
        "initial_calibration_support": initial_support,
        "initial_entropy_bits": initial["entropy_bits"],
        "initial_map_abs_error_um": initial["map_abs_error_um"],
        "initial_posterior_mean_abs_error_um": initial["posterior_mean_abs_error_um"],
        **{f"final_{key}": value for key, value in final.items()},
        "selected_actions_um": selected_offsets,
        "selected_action_count": len(selected_offsets),
        "unique_action_count": len(set(selected_offsets)),
        "sum_predicted_information_gain": float(
            sum(float(item["information_gain"]) for item in trace)
        ),
        "minimum_calibration_support": int(
            min([initial_support] + [int(item["calibration_support_after"]) for item in trace])
        ),
        "entropy_reduction_bits": float(initial["entropy_bits"] - final["entropy_bits"]),
        "initial_residual_um": float(initial_residual),
        "best_acquired_residual_um": float(best_acquired_residual),
        "last_acquired_residual_um": float(last_acquired_residual),
        "oracle_best_one_action_residual_um": float(available_oracle),
        "best_view_regret_um": float(best_acquired_residual - available_oracle),
        "trace": trace,
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"], int(row["budget"])].append(row)
    result = []
    for (policy, budget), values in sorted(grouped.items()):
        actions = [action for row in values for action in row["selected_actions_um"]]
        counts = Counter(actions)
        result.append(
            {
                "policy": policy,
                "budget": budget,
                "episodes": len(values),
                "mean_map_abs_error_um": float(
                    np.mean([row["final_map_abs_error_um"] for row in values])
                ),
                "mean_posterior_mean_abs_error_um": float(
                    np.mean([row["final_posterior_mean_abs_error_um"] for row in values])
                ),
                "mean_entropy_bits": float(np.mean([row["final_entropy_bits"] for row in values])),
                "mean_entropy_reduction_bits": float(
                    np.mean([row["entropy_reduction_bits"] for row in values])
                ),
                "mean_true_state_probability": float(
                    np.mean([row["final_true_state_probability"] for row in values])
                ),
                "within_6um_fraction": float(np.mean([row["final_within_6um"] for row in values])),
                "within_12um_fraction": float(
                    np.mean([row["final_within_12um"] for row in values])
                ),
                "mean_best_acquired_residual_um": float(
                    np.mean([row["best_acquired_residual_um"] for row in values])
                ),
                "mean_best_view_regret_um": float(
                    np.mean([row["best_view_regret_um"] for row in values])
                ),
                "mean_predicted_information_gain": float(
                    np.mean([row["sum_predicted_information_gain"] for row in values])
                ),
                "mean_minimum_calibration_support": float(
                    np.mean([row["minimum_calibration_support"] for row in values])
                ),
                "zero_support_fraction": float(
                    np.mean([row["minimum_calibration_support"] == 0 for row in values])
                ),
                "distinct_actions_selected": len(counts),
                "action_counts": dict(sorted(counts.items())),
            }
        )
    return result


def group_bootstrap_delta(
    rows: list[dict],
    *,
    budget: int,
    baseline_policy: str,
    metric: str,
    seed: int,
    repeats: int,
) -> dict:
    selected = [
        row
        for row in rows
        if int(row["budget"]) == budget and row["policy"] in {"eig", baseline_policy}
    ]
    by_key = {
        (row["specimen_id"], int(row["true_state_um"]), row["policy"]): row for row in selected
    }
    specimens = sorted({row["specimen_id"] for row in selected})
    if not specimens:
        raise ValueError("no specimens for bootstrap comparison")

    def observed_for(specimen_sample: list[str]) -> float:
        eig_values = []
        base_values = []
        for specimen in specimen_sample:
            states = [
                state
                for state in STATE_OFFSETS_UM
                if (specimen, state, "eig") in by_key
                and (specimen, state, baseline_policy) in by_key
            ]
            for state in states:
                eig_values.append(float(by_key[specimen, state, "eig"][metric]))
                base_values.append(float(by_key[specimen, state, baseline_policy][metric]))
        if not eig_values:
            raise ValueError("paired bootstrap has no episodes")
        return float(np.mean(eig_values) - np.mean(base_values))

    observed = observed_for(specimens)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sample = rng.choice(specimens, size=len(specimens), replace=True).tolist()
        draws[i] = observed_for(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "budget": budget,
        "baseline_policy": baseline_policy,
        "metric": metric,
        "eig_minus_baseline": observed,
        "ci95": [float(low), float(high)],
        "bootstrap_unit": "specimen",
        "bootstrap_repeats": repeats,
    }


def render_report(
    stage: str,
    summaries: list[dict],
    comparisons: list[dict],
    split: dict[str, tuple[str, ...]],
) -> str:
    lines = [
        f"# LSFM Bayesian Replay — {stage.title()}",
        "",
        "**Optical engineering only — not biological validation.**",
        "",
        "## Split",
        "",
        f"- Calibration specimens: **{len(split['calibration'])}**",
        f"- Development specimens: **{len(split['development'])}**",
        f"- Test specimens: **{len(split['test'])}**",
        "",
        "## Policy summary",
        "",
        "| Policy | Budget | MAP MAE | Posterior-mean MAE | Entropy | Within 6um | Best acquired residual | Distinct actions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['policy']} | {row['budget']}"
            f" | {row['mean_map_abs_error_um']:.3f}"
            f" | {row['mean_posterior_mean_abs_error_um']:.3f}"
            f" | {row['mean_entropy_bits']:.3f}"
            f" | {row['within_6um_fraction']:.3f}"
            f" | {row['mean_best_acquired_residual_um']:.3f}"
            f" | {row['distinct_actions_selected']} |"
        )
    lines += [
        "",
        "## Paired EIG comparisons",
        "",
        "Negative delta favors EIG for MAE and entropy.",
        "",
        "| Budget | Baseline | Metric | EIG - baseline | 95% specimen bootstrap CI |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['budget']} | {row['baseline_policy']} | {row['metric']}"
            f" | {row['eig_minus_baseline']:+.4f}"
            f" | [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This benchmark tests multi-action optical-state inference and action selection.",
        "It does not test the biology-vs-nuisance verification claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-matrix", required=True)
    parser.add_argument("--stage", choices=["development", "test"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()

    source = Path(args.response_matrix).resolve()
    out = Path(args.out).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if out.exists():
        raise FileExistsError(f"refusing overwrite: {out}")
    if args.bootstrap_repeats < 200:
        raise ValueError("bootstrap repeats must be >= 200")

    rows = read_response_matrix(source)
    episodes = build_episode_index(rows)
    specimens = sorted({key[0] for key in episodes})
    split = split_specimens(specimens, args.seed)

    model, model_report = fit_model(
        episodes,
        split["calibration"],
        n_bins=args.n_bins,
        seed=args.seed,
        alpha=args.alpha,
    )

    evaluation_specimens = set(split[args.stage])
    episode_rows = []
    for specimen, true_state in sorted(episodes):
        if specimen not in evaluation_specimens:
            continue
        for budget in (1, 2, 3):
            for policy in POLICIES:
                episode_rows.append(
                    run_episode(
                        model,
                        specimen,
                        true_state,
                        episodes[specimen, true_state],
                        policy,
                        budget,
                        args.seed,
                    )
                )

    expected = len(evaluation_specimens) * len(STATE_OFFSETS_UM) * 3 * len(POLICIES)
    if len(episode_rows) != expected:
        raise ValueError(f"episode-count mismatch: expected {expected}, got {len(episode_rows)}")

    summaries = summarize(episode_rows)
    comparisons = []
    baselines = [policy for policy in POLICIES if policy not in {"eig", "oracle_focus"}]
    for budget in (1, 2, 3):
        for baseline in baselines:
            for metric in (
                "final_posterior_mean_abs_error_um",
                "final_entropy_bits",
            ):
                comparisons.append(
                    group_bootstrap_delta(
                        episode_rows,
                        budget=budget,
                        baseline_policy=baseline,
                        metric=metric,
                        seed=stable_seed(args.seed, args.stage, budget, baseline, metric),
                        repeats=args.bootstrap_repeats,
                    )
                )

    out.mkdir(parents=True)
    write_csv(out / "episodes.csv", episode_rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "paired_comparisons.csv", comparisons)
    atomic_json(out / "model_report.json", model_report)
    atomic_json(
        out / "split.json",
        {name: list(values) for name, values in split.items()},
    )
    run = {
        "state": "COMPLETE",
        "stage": args.stage,
        "source_response_matrix": str(source),
        "source_response_matrix_sha256": digest(source),
        "seed": args.seed,
        "n_bins": args.n_bins,
        "alpha": args.alpha,
        "bootstrap_repeats": args.bootstrap_repeats,
        "calibration_specimens": len(split["calibration"]),
        "evaluation_specimens": len(evaluation_specimens),
        "episodes": len(episode_rows),
        "policies": list(POLICIES),
        "budgets": [1, 2, 3],
        "biological_validation": False,
        "hardware_validation": False,
        "environment": environment(),
    }
    atomic_json(out / "run.json", run)
    report = render_report(args.stage, summaries, comparisons, split)
    (out / "REPORT.md").write_text(report + "\n", encoding="utf-8")

    artifacts = [
        "episodes.csv",
        "summary.csv",
        "paired_comparisons.csv",
        "model_report.json",
        "split.json",
        "run.json",
        "REPORT.md",
    ]
    atomic_json(
        out / "artifact_hashes.json",
        {name: digest(out / name) for name in artifacts},
    )

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
