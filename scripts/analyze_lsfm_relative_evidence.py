"""Leakage-safe LSFM relative-action optical evidence diagnostic."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from counterfactual_microscopy.research.datasets import read_image
from counterfactual_microscopy.research.images import (
    FEATURE_NAMES,
    features,
    quality,
    register,
)
from counterfactual_microscopy.research.io import (
    atomic_json,
    digest,
    environment,
    write_csv,
)

START_OFFSETS_UM = (
    -30,
    -24,
    -18,
    -12,
    -6,
    6,
    12,
    18,
    24,
    30,
)

ACTION_OFFSETS_UM = (
    -18,
    -12,
    -6,
    6,
    12,
    18,
)

BASE_FEATURES = (
    "baseline_quality",
    "baseline_std",
    "baseline_gradient",
    "baseline_laplacian",
    "baseline_maxima",
    "baseline_bright_fraction",
    "baseline_dark_fraction",
)

PAIR_FEATURES = BASE_FEATURES + (
    "quality_delta",
    "std_delta",
    "gradient_delta",
    "laplacian_delta",
    "maxima_delta",
    "bright_fraction_delta",
    "dark_fraction_delta",
    "one_minus_correlation",
    "registration_shift",
)


def offset_to_plane(offset_um: float) -> int:
    half_steps = float(offset_um) / 2.0

    if not half_steps.is_integer():
        raise ValueError(f"offset {offset_um} is not on the 2-um LSFM grid")

    plane = 26 + int(half_steps)

    if not 1 <= plane <= 51:
        raise ValueError(f"offset {offset_um} lies outside the LSFM stack")

    return plane


def required_planes() -> tuple[int, ...]:
    offsets = set(START_OFFSETS_UM)

    offsets.update(start + action for start in START_OFFSETS_UM for action in ACTION_OFFSETS_UM)

    return tuple(sorted(offset_to_plane(offset) for offset in offsets))


def descriptor(image: np.ndarray) -> dict[str, float]:
    values = features(image[None])[0]

    lookup = {name: index for index, name in enumerate(FEATURE_NAMES)}

    return {
        "quality": quality(image),
        "std": float(values[lookup["std"]]),
        "gradient": float(values[lookup["gradient"]]),
        "laplacian": float(values[lookup["laplacian"]]),
        "maxima": float(values[lookup["maxima"]]),
        "bright_fraction": float(values[lookup["bright_fraction"]]),
        "dark_fraction": float(values[lookup["dark_fraction"]]),
    }


def read_manifest(
    manifest: Path,
) -> tuple[
    list[str],
    dict[tuple[str, int], dict[str, str]],
]:
    with manifest.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "specimen_id",
        "plane",
        "path",
        "sha256",
        "pairing_verified",
        "focus_plane",
        "step_um",
        "biological_label_available",
    }

    if not rows:
        raise ValueError("empty LSFM manifest")

    if not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0]))

        raise ValueError(f"manifest missing fields: {missing}")

    if {int(row["focus_plane"]) for row in rows} != {26}:
        raise ValueError("LSFM manifest focus plane must be 26")

    if {float(row["step_um"]) for row in rows} != {2.0}:
        raise ValueError("LSFM manifest spacing must be 2 um")

    if any(row["pairing_verified"] not in {"1", "true", "True"} for row in rows):
        raise ValueError("manifest contains unverified pairing")

    if any(row["biological_label_available"] not in {"0", "false", "False"} for row in rows):
        raise ValueError("LSFM engineering data unexpectedly contains biological labels")

    views: dict[
        tuple[str, int],
        dict[str, str],
    ] = {}

    for row in rows:
        key = (
            row["specimen_id"],
            int(row["plane"]),
        )

        if key in views:
            raise ValueError(f"duplicate specimen-plane entry: {key}")

        views[key] = row

    specimens = sorted({row["specimen_id"] for row in rows})

    if len(specimens) != 42:
        raise ValueError(f"expected 42 specimens, found {len(specimens)}")

    expected_planes = list(range(1, 52))

    for specimen in specimens:
        observed = sorted(plane for specimen_id, plane in views if specimen_id == specimen)

        if observed != expected_planes:
            raise ValueError(f"incomplete stack for {specimen}")

    return specimens, views


def resolve_path(
    manifest: Path,
    stored_path: str,
) -> Path:
    path = Path(stored_path)

    if not path.is_absolute():
        path = manifest.parent / path

    path = path.resolve()

    if not path.is_file():
        raise ValueError(f"missing LSFM TIFF: {path}")

    return path


def load_cache(
    manifest: Path,
    specimens: list[str],
    views: dict[
        tuple[str, int],
        dict[str, str],
    ],
    *,
    size: int,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    dict[
        tuple[str, int],
        dict[str, float],
    ],
]:
    image_cache: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}

    descriptor_cache: dict[
        tuple[str, int],
        dict[str, float],
    ] = {}

    planes = required_planes()

    total = len(specimens) * len(planes)
    loaded = 0

    for specimen in specimens:
        for plane in planes:
            row = views[
                specimen,
                plane,
            ]

            path = resolve_path(
                manifest,
                row["path"],
            )

            actual_hash = digest(path)

            if actual_hash != row["sha256"]:
                raise ValueError(f"image changed since manifest creation: {path}")

            image = read_image(
                path,
                size,
            )

            key = (
                specimen,
                plane,
            )

            image_cache[key] = image
            descriptor_cache[key] = descriptor(image)

            loaded += 1

        print(f"Loaded {loaded}/{total} required images")

    return image_cache, descriptor_cache


def build_rows(
    specimens: list[str],
    images: dict[
        tuple[str, int],
        np.ndarray,
    ],
    descriptors: dict[
        tuple[str, int],
        dict[str, float],
    ],
) -> tuple[list[dict], list[dict]]:
    baseline_rows = []
    pair_rows = []

    for specimen in specimens:
        for start_offset in START_OFFSETS_UM:
            baseline_plane = offset_to_plane(start_offset)

            baseline = images[
                specimen,
                baseline_plane,
            ]

            base = descriptors[
                specimen,
                baseline_plane,
            ]

            baseline_record = {
                "specimen_id": specimen,
                "start_offset_um": start_offset,
                "start_sign_positive": int(start_offset > 0),
                "baseline_plane": baseline_plane,
                "baseline_quality": base["quality"],
                "baseline_std": base["std"],
                "baseline_gradient": base["gradient"],
                "baseline_laplacian": base["laplacian"],
                "baseline_maxima": base["maxima"],
                "baseline_bright_fraction": base["bright_fraction"],
                "baseline_dark_fraction": base["dark_fraction"],
            }

            baseline_rows.append(baseline_record)

            for action_offset in ACTION_OFFSETS_UM:
                candidate_offset = start_offset + action_offset

                candidate_plane = offset_to_plane(candidate_offset)

                candidate = images[
                    specimen,
                    candidate_plane,
                ]

                cand = descriptors[
                    specimen,
                    candidate_plane,
                ]

                registration = register(
                    baseline,
                    candidate,
                    max_shift=32.0,
                    min_correlation=-1.0,
                    min_overlap=0.5,
                )

                initial_distance = abs(start_offset)

                final_distance = abs(candidate_offset)

                improvement = initial_distance - final_distance

                pair_rows.append(
                    {
                        **baseline_record,
                        "action_offset_um": action_offset,
                        "candidate_offset_um": candidate_offset,
                        "candidate_plane": candidate_plane,
                        "focus_improvement_um": improvement,
                        "moves_toward_focus": int(improvement > 0),
                        "candidate_quality": cand["quality"],
                        "quality_delta": (cand["quality"] - base["quality"]),
                        "std_delta": (cand["std"] - base["std"]),
                        "gradient_delta": (cand["gradient"] - base["gradient"]),
                        "laplacian_delta": (cand["laplacian"] - base["laplacian"]),
                        "maxima_delta": (cand["maxima"] - base["maxima"]),
                        "bright_fraction_delta": (
                            cand["bright_fraction"] - base["bright_fraction"]
                        ),
                        "dark_fraction_delta": (cand["dark_fraction"] - base["dark_fraction"]),
                        "correlation": (registration.correlation),
                        "one_minus_correlation": (1.0 - registration.correlation),
                        "registration_shift": float(
                            np.hypot(
                                registration.shift_y,
                                registration.shift_x,
                            )
                        ),
                        "registration_overlap": (registration.overlap),
                        "registration_valid": int(registration.valid),
                        "registration_reason": (registration.reason),
                    }
                )

    if len(baseline_rows) != 420:
        raise ValueError(f"baseline-state count mismatch: {len(baseline_rows)}")

    if len(pair_rows) != 2520:
        raise ValueError(f"response-pair count mismatch: {len(pair_rows)}")

    return baseline_rows, pair_rows


def specimen_folds(
    specimens: list[str],
    seed: int,
) -> list[set[str]]:
    if len(specimens) != 42:
        raise ValueError("fold builder expects 42 specimens")

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(len(specimens))

    folds = []

    for indices in np.array_split(
        permutation,
        3,
    ):
        folds.append({specimens[int(index)] for index in indices})

    union = set().union(*folds)

    if union != set(specimens):
        raise ValueError("folds do not cover all specimens")

    if sum(len(fold) for fold in folds) != len(union):
        raise ValueError("specimen leakage between folds")

    if sorted(len(fold) for fold in folds) != [
        14,
        14,
        14,
    ]:
        raise ValueError("unexpected fold sizes")

    return folds


def feature_matrix(
    rows: list[dict],
    columns: tuple[str, ...],
) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError("feature matrix is not two-dimensional")

    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains nonfinite values")

    return matrix


def evaluate_decoder(
    rows: list[dict],
    columns: tuple[str, ...],
    folds: list[set[str]],
) -> list[dict]:
    metrics = []

    for fold_index, held_out in enumerate(folds):
        train_rows = [row for row in rows if row["specimen_id"] not in held_out]

        test_rows = [row for row in rows if row["specimen_id"] in held_out]

        if not train_rows or not test_rows:
            raise ValueError("empty train/test fold")

        train_specimens = {row["specimen_id"] for row in train_rows}

        test_specimens = {row["specimen_id"] for row in test_rows}

        if train_specimens & test_specimens:
            raise ValueError("specimen leakage detected")

        x_train = feature_matrix(
            train_rows,
            columns,
        )

        x_test = feature_matrix(
            test_rows,
            columns,
        )

        sign_train = np.asarray(
            [int(row["start_sign_positive"]) for row in train_rows],
            dtype=int,
        )

        sign_test = np.asarray(
            [int(row["start_sign_positive"]) for row in test_rows],
            dtype=int,
        )

        offset_train = np.asarray(
            [float(row["start_offset_um"]) for row in train_rows],
            dtype=float,
        )

        offset_test = np.asarray(
            [float(row["start_offset_um"]) for row in test_rows],
            dtype=float,
        )

        if len(np.unique(sign_train)) != 2:
            raise ValueError("training fold lacks both sign classes")

        if len(np.unique(sign_test)) != 2:
            raise ValueError("test fold lacks both sign classes")

        sign_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                random_state=0,
            ),
        )

        offset_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )

        sign_model.fit(
            x_train,
            sign_train,
        )

        offset_model.fit(
            x_train,
            offset_train,
        )

        sign_probability = sign_model.predict_proba(x_test)[:, 1]

        sign_prediction = (sign_probability >= 0.5).astype(int)

        offset_prediction = offset_model.predict(x_test)

        metrics.append(
            {
                "fold": fold_index,
                "train_specimens": len(train_specimens),
                "test_specimens": len(test_specimens),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "sign_roc_auc": float(
                    roc_auc_score(
                        sign_test,
                        sign_probability,
                    )
                ),
                "sign_accuracy": float(
                    accuracy_score(
                        sign_test,
                        sign_prediction,
                    )
                ),
                "offset_mae_um": float(
                    mean_absolute_error(
                        offset_test,
                        offset_prediction,
                    )
                ),
            }
        )

    return metrics


def summarize_folds(
    fold_metrics: list[dict],
) -> dict:
    if len(fold_metrics) != 3:
        raise ValueError("expected exactly three CV folds")

    return {
        "mean_sign_roc_auc": float(np.mean([row["sign_roc_auc"] for row in fold_metrics])),
        "mean_sign_accuracy": float(np.mean([row["sign_accuracy"] for row in fold_metrics])),
        "mean_offset_mae_um": float(np.mean([row["offset_mae_um"] for row in fold_metrics])),
        "folds": fold_metrics,
    }


def evaluate_actions(
    pair_rows: list[dict],
    baseline: dict,
    folds: list[set[str]],
) -> list[dict]:
    output = []

    baseline_auc = float(baseline["mean_sign_roc_auc"])

    baseline_mae = float(baseline["mean_offset_mae_um"])

    for action_offset in ACTION_OFFSETS_UM:
        rows = [row for row in pair_rows if int(row["action_offset_um"]) == action_offset]

        if len(rows) != 420:
            raise ValueError(f"action {action_offset} has {len(rows)} rows instead of 420")

        pair_summary = summarize_folds(
            evaluate_decoder(
                rows,
                PAIR_FEATURES,
                folds,
            )
        )

        nonneutral = [row for row in rows if float(row["focus_improvement_um"]) != 0.0]

        if not nonneutral:
            raise ValueError("action has no non-neutral cases")

        quality_direction_accuracy = float(
            np.mean(
                [
                    (float(row["quality_delta"]) > 0) == (float(row["focus_improvement_um"]) > 0)
                    for row in nonneutral
                ]
            )
        )

        registration_valid_rate = float(np.mean([int(row["registration_valid"]) for row in rows]))

        pair_auc = float(pair_summary["mean_sign_roc_auc"])

        pair_mae = float(pair_summary["mean_offset_mae_um"])

        auc_gain = pair_auc - baseline_auc

        mae_reduction_um = baseline_mae - pair_mae

        if baseline_mae <= 0:
            raise ValueError("baseline MAE must be positive")

        mae_reduction_fraction = mae_reduction_um / baseline_mae

        passes = pair_auc >= 0.70 and auc_gain >= 0.05 and mae_reduction_fraction >= 0.10

        output.append(
            {
                "action_offset_um": (action_offset),
                "rows": len(rows),
                "pair_sign_roc_auc": (pair_auc),
                "pair_sign_accuracy": float(pair_summary["mean_sign_accuracy"]),
                "pair_offset_mae_um": (pair_mae),
                "baseline_sign_roc_auc": (baseline_auc),
                "baseline_offset_mae_um": (baseline_mae),
                "auc_gain_over_baseline": (auc_gain),
                "mae_reduction_um": (mae_reduction_um),
                "mae_reduction_fraction": (mae_reduction_fraction),
                "quality_direction_accuracy": (quality_direction_accuracy),
                "registration_valid_rate": (registration_valid_rate),
                "passes_incremental_gate": (passes),
                "folds": (pair_summary["folds"]),
            }
        )

    return output


def render_report(
    baseline: dict,
    actions: list[dict],
    folds: list[set[str]],
) -> tuple[str, bool]:
    passing_actions = [row for row in actions if row["passes_incremental_gate"]]

    mean_pair_auc = float(np.mean([float(row["pair_sign_roc_auc"]) for row in actions]))

    proceed = len(passing_actions) >= 2 and mean_pair_auc >= 0.70

    lines = [
        "# LSFM Relative-Action Evidence Diagnostic",
        "",
        ("**Optical engineering diagnostic only — not biological or hardware validation.**"),
        "",
        "## Baseline-image-only decoder",
        "",
        (f"- Mean signed-defocus ROC AUC: **{baseline['mean_sign_roc_auc']:.4f}**"),
        (f"- Mean sign accuracy: **{baseline['mean_sign_accuracy']:.4f}**"),
        (f"- Mean signed-offset MAE: **{baseline['mean_offset_mae_um']:.3f} um**"),
        "",
        "## Relative actions",
        "",
        (
            "| Move | Pair AUC | AUC gain | "
            "Offset MAE | MAE reduction | "
            "Quality direction | Registration valid | Gate |"
        ),
        ("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"),
    ]

    for row in actions:
        lines.append(
            f"| {int(row['action_offset_um']):+d} um"
            f" | {float(row['pair_sign_roc_auc']):.4f}"
            f" | {float(row['auc_gain_over_baseline']):+.4f}"
            f" | {float(row['pair_offset_mae_um']):.3f}"
            f" | {100 * float(row['mae_reduction_fraction']):+.1f}%"
            f" | {float(row['quality_direction_accuracy']):.4f}"
            f" | {float(row['registration_valid_rate']):.4f}"
            f" | {bool(row['passes_incremental_gate'])} |"
        )

    lines += [
        "",
        "## Prespecified continuation gate",
        "",
        (f"- Incrementally informative actions: **{len(passing_actions)} / 6**"),
        (f"- Mean pair AUC: **{mean_pair_auc:.4f}**"),
        (f"- Proceed to Bayesian/EIG replay: **{proceed}**"),
        "",
        "## CV folds",
        "",
    ]

    for index, fold in enumerate(folds):
        lines.append(f"- Fold {index}: {len(fold)} held-out specimens")

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "A high pair AUC alone is insufficient. "
            "The pair must improve over baseline-only decoding."
        ),
        "",
        ("No pair-level p-value is interpreted as if 2,520 response pairs were independent."),
        "",
    ]

    return "\n".join(lines), proceed


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260911,
    )

    args = parser.parse_args()

    if args.size < 16:
        raise ValueError("image size must be at least 16")

    manifest = Path(args.manifest).resolve()

    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")

    out = Path(args.out).resolve()

    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")

    out.mkdir(parents=True)

    specimens, views = read_manifest(manifest)

    folds = specimen_folds(
        specimens,
        args.seed,
    )

    images, descriptors = load_cache(
        manifest,
        specimens,
        views,
        size=args.size,
    )

    baseline_rows, pair_rows = build_rows(
        specimens,
        images,
        descriptors,
    )

    baseline_folds = evaluate_decoder(
        baseline_rows,
        BASE_FEATURES,
        folds,
    )

    baseline = summarize_folds(baseline_folds)

    actions = evaluate_actions(
        pair_rows,
        baseline,
        folds,
    )

    report, proceed = render_report(
        baseline,
        actions,
        folds,
    )

    write_csv(
        out / "baseline_states.csv",
        baseline_rows,
    )

    write_csv(
        out / "response_matrix.csv",
        pair_rows,
    )

    flat_action_rows = [
        {key: value for key, value in row.items() if key != "folds"} for row in actions
    ]

    write_csv(
        out / "action_diagnostics.csv",
        flat_action_rows,
    )

    atomic_json(
        out / "baseline_decoder.json",
        baseline,
    )

    atomic_json(
        out / "action_diagnostics.json",
        actions,
    )

    atomic_json(
        out / "folds.json",
        {f"fold_{index}": sorted(fold) for index, fold in enumerate(folds)},
    )

    atomic_json(
        out / "run.json",
        {
            "state": "COMPLETE",
            "stage": ("lsfm_relative_evidence"),
            "manifest": str(manifest),
            "manifest_sha256": digest(manifest),
            "seed": args.seed,
            "image_size": args.size,
            "specimens": len(specimens),
            "baseline_states": len(baseline_rows),
            "response_pairs": len(pair_rows),
            "required_planes": (required_planes()),
            "start_offsets_um": (START_OFFSETS_UM),
            "action_offsets_um": (ACTION_OFFSETS_UM),
            "proceed_to_eig": (proceed),
            "biological_validation": False,
            "hardware_validation": False,
            "environment": environment(),
        },
    )

    (out / "REPORT.md").write_text(
        report + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "baseline_states.csv",
        "response_matrix.csv",
        "action_diagnostics.csv",
        "baseline_decoder.json",
        "action_diagnostics.json",
        "folds.json",
        "run.json",
        "REPORT.md",
    ]

    atomic_json(
        out / "artifact_hashes.json",
        {name: digest(out / name) for name in artifact_names},
    )

    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
