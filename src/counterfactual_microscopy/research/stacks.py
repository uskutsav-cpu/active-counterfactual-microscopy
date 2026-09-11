"""Audited same-specimen optical-stack indexing.

This module intentionally keeps optical-stack geometry separate from biological
task labels. A focus stack can validate intervention mechanics without being
misrepresented as biological validation.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import atomic_json, digest, write_csv
from .validation import finite, integer

LSFM_FILENAME = re.compile(
    r"^(?P<specimen>.+)_(?P<plane>\d{2})\.tiff?$",
    re.IGNORECASE,
)

# The authors' public autofocus loader uses these levels when the distance
# between adjacent model classes is 6 um. Plane 26 is the in-focus reference.
LSFM_PUBLISHED_6UM_PLANES = (
    8,
    11,
    14,
    17,
    20,
    23,
    26,
    29,
    32,
    35,
    38,
    41,
    44,
)


def plane_offset_um(
    plane: int,
    *,
    focus_plane: int = 26,
    step_um: float = 2.0,
) -> float:
    """Convert one indexed stack plane into signed axial defocus."""
    integer("plane", plane, 1)
    integer("focus_plane", focus_plane, 1)
    finite("step_um", step_um, 0)

    if step_um <= 0:
        raise ValueError("step_um must be positive")

    return float((plane - focus_plane) * step_um)


def action_name(offset_um: float) -> str:
    """Stable Intervention-compatible name for a signed focus displacement."""
    finite("offset_um", offset_um)

    rounded = round(float(offset_um), 8)

    if rounded == 0:
        return "focus_0um"

    magnitude = abs(rounded)

    if float(magnitude).is_integer():
        token = str(int(magnitude))
    else:
        token = str(magnitude).replace(".", "p")

    direction = "p" if rounded > 0 else "m"

    return f"focus_{direction}{token}um"


def _parse_lsfm_path(path: Path) -> tuple[str, int]:
    """Return specimen identity and plane, validating folder/file agreement."""
    match = LSFM_FILENAME.match(path.name)

    if not match:
        raise ValueError(f"unrecognized LSFM filename: {path.name}")

    try:
        folder_plane = int(path.parent.name)
    except ValueError as exc:
        raise ValueError(f"LSFM TIFF is not inside a numeric plane folder: {path}") from exc

    suffix_plane = int(match.group("plane"))

    if folder_plane != suffix_plane:
        raise ValueError(
            f"folder/file plane disagreement: {path} (folder={folder_plane}, suffix={suffix_plane})"
        )

    return match.group("specimen"), folder_plane


def index_lsfm(
    root: str | Path,
    out: str | Path,
    *,
    focus_plane: int = 26,
    step_um: float = 2.0,
    expected_planes: int = 51,
) -> dict:
    """Create a hash-audited same-specimen LSFM stack manifest.

    The manifest contains optical geometry only. It does not invent a
    biological label.
    """
    integer("focus_plane", focus_plane, 1)
    integer("expected_planes", expected_planes, 1)
    finite("step_um", step_um, 0)

    if step_um <= 0:
        raise ValueError("step_um must be positive")

    root = Path(root).resolve()
    out = Path(out).resolve()

    if out.exists():
        raise FileExistsError(f"refusing overwrite {out}")

    if not root.is_dir():
        raise ValueError(f"stack root does not exist: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )

    if not paths:
        raise ValueError("no TIFF images found")

    rows: list[dict] = []
    errors: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for path in paths:
        try:
            raw_specimen, plane = _parse_lsfm_path(path)

            key = (raw_specimen, plane)

            if key in seen:
                raise ValueError(f"duplicate specimen/plane: {raw_specimen}, {plane}")

            seen.add(key)

            offset = plane_offset_um(
                plane,
                focus_plane=focus_plane,
                step_um=step_um,
            )

            rows.append(
                {
                    "dataset": "lsfm_defocus",
                    "specimen_id": f"lsfm_{raw_specimen}",
                    "group_id": f"lsfm_{raw_specimen}",
                    "raw_specimen_id": raw_specimen,
                    "plane": plane,
                    "condition": f"plane_{plane:02d}",
                    "offset_um": offset,
                    "is_reference_focus": int(plane == focus_plane),
                    "path": str(path),
                    "sha256": digest(path),
                    "pairing_verified": 1,
                    "step_um": step_um,
                    "focus_plane": focus_plane,
                    "biological_label_available": 0,
                }
            )

        except ValueError as exc:
            errors.append(
                {
                    "path": str(path),
                    "error": str(exc),
                }
            )

    if errors:
        atomic_json(
            out.with_suffix(".index-errors.json"),
            errors,
        )
        raise ValueError(f"{len(errors)} LSFM files failed indexing; none were silently omitted")

    by_specimen: dict[str, dict[int, dict]] = defaultdict(dict)

    for row in rows:
        specimen = row["specimen_id"]
        plane = int(row["plane"])

        if plane in by_specimen[specimen]:
            raise ValueError("duplicate LSFM plane after indexing")

        by_specimen[specimen][plane] = row

    planes = sorted({int(row["plane"]) for row in rows})

    expected = list(range(1, expected_planes + 1))

    if planes != expected:
        raise ValueError(f"unexpected LSFM plane set: expected {expected}, got {planes}")

    if focus_plane not in planes:
        raise ValueError("reference focus plane is absent")

    incomplete: dict[str, list[int]] = {}

    for specimen, views in sorted(by_specimen.items()):
        missing = sorted(set(planes) - set(views))

        if missing:
            incomplete[specimen] = missing

    if incomplete:
        atomic_json(
            out.with_suffix(".incomplete.json"),
            incomplete,
        )
        raise ValueError(f"{len(incomplete)} incomplete LSFM stacks")

    per_plane = Counter(int(row["plane"]) for row in rows)

    if len(set(per_plane.values())) != 1:
        raise ValueError("plane folders contain unequal specimen counts")

    write_csv(out, rows)

    offsets = [float(row["offset_um"]) for row in rows]

    report = {
        "dataset": "lsfm_defocus",
        "images": len(rows),
        "specimens": len(by_specimen),
        "groups": len(by_specimen),
        "planes": len(planes),
        "plane_min": min(planes),
        "plane_max": max(planes),
        "focus_plane": focus_plane,
        "step_um": step_um,
        "offset_min_um": min(offsets),
        "offset_max_um": max(offsets),
        "images_per_plane": dict(sorted(per_plane.items())),
        "complete_stacks": len(by_specimen),
        "incomplete_stacks": 0,
        "pairing_verified": True,
        "biological_label_available": False,
        "manifest_sha256": digest(out),
    }

    atomic_json(
        out.with_suffix(".audit.json"),
        report,
    )

    return report


def action_catalog(
    manifest: str | Path,
    out: str | Path,
    *,
    planes: list[int] | tuple[int, ...] | None = None,
) -> dict:
    """Freeze a candidate optical-action catalog from a stack manifest."""
    manifest = Path(manifest).resolve()
    out = Path(out).resolve()

    if out.exists():
        raise FileExistsError(f"refusing overwrite {out}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "specimen_id",
        "plane",
        "offset_um",
        "step_um",
        "focus_plane",
        "pairing_verified",
    }

    if not rows or not required.issubset(rows[0]):
        raise ValueError("stack manifest missing required columns")

    focus_values = {int(row["focus_plane"]) for row in rows}

    step_values = {float(row["step_um"]) for row in rows}

    if len(focus_values) != 1 or len(step_values) != 1:
        raise ValueError("inconsistent stack geometry")

    focus_plane = next(iter(focus_values))
    step_um = next(iter(step_values))

    requested = tuple(LSFM_PUBLISHED_6UM_PLANES if planes is None else planes)

    if not requested or len(set(requested)) != len(requested):
        raise ValueError("action planes must be nonempty and unique")

    available_by_specimen: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        if row["pairing_verified"] not in {"1", "true", "True"}:
            raise ValueError("manifest contains unverified pairing")

        available_by_specimen[row["specimen_id"]].add(int(row["plane"]))

    missing: dict[str, list[int]] = {}

    for specimen, available in sorted(available_by_specimen.items()):
        absent = [plane for plane in requested if plane not in available]

        if absent:
            missing[specimen] = absent

    if missing:
        raise ValueError(f"{len(missing)} specimens lack requested action planes")

    actions = []

    for plane in requested:
        offset = plane_offset_um(
            plane,
            focus_plane=focus_plane,
            step_um=step_um,
        )

        actions.append(
            {
                "name": action_name(offset),
                "plane": int(plane),
                "offset_um": offset,
                "distance_from_focus_um": abs(offset),
                "is_reference_focus": plane == focus_plane,
                "dose_proxy": 1.0,
                "time_proxy": 1.0,
                "physical_cost_calibrated": False,
            }
        )

    report = {
        "source_manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "specimens": len(available_by_specimen),
        "focus_plane": focus_plane,
        "step_um": step_um,
        "action_count": len(actions),
        "actions": actions,
        "biological_validation": False,
        "hardware_validation": False,
        "purpose": ("engineering validation of a multi-action same-specimen optical replay space"),
    }

    atomic_json(out, report)

    return report
