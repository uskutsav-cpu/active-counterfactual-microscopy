"""BBBC005/006 indexing and generic manifest-to-panel conversion.

BBBC006 count labels are algorithm-generated at z16, not independent human counts.
BBBC005 count/sample/channel keys are only pairing CANDIDATES: replay requires an
explicit pairing confirmation. Identical counts do not establish specimen identity.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from .images import as_float
from .io import atomic_json, digest, save_panel, seed_for, write_csv
from .splits import assert_disjoint, indices, record
from .types import Intervention, Panel
from .validation import finite, integer

BBBC005 = re.compile(
    "SIMCEPImages_(?P<well>[A-P]\\d{2})_C(?P<count>\\d+)_F(?P<focus>\\d+)_s(?P<site>\\d+)_w(?P<channel>[12])\\.tiff?$",
    re.IGNORECASE,
)
BBBC006 = re.compile(
    "_(?P<well>[a-p]\\d{2})_s(?P<site>\\d+)_w(?P<channel>[12])[^/]*\\.tiff?$", re.IGNORECASE
)


def parse_bbbc005(path: str | Path) -> dict:
    m = BBBC005.search(Path(path).name)
    if not m:
        raise ValueError(f"not a recognized BBBC005 image: {path}")
    d = m.groupdict()
    count = int(d["count"])
    focus = int(d["focus"])
    site = int(d["site"])
    channel = int(d["channel"])
    if not 1 <= count <= 100 or not 1 <= focus <= 48:
        raise ValueError("BBBC005 metadata out of bounds")
    return {
        "specimen_id": f"bbbc005_c{count:03d}_s{site:02d}_w{channel}",
        "group_id": f"bbbc005_c{count:03d}_s{site:02d}",
        "condition": f"f{focus:02d}",
        "task_value": count,
        "channel": channel,
        "well": d["well"].lower(),
    }


def parse_bbbc006(path: str | Path) -> dict:
    p = Path(path)
    m = BBBC006.search(p.name)
    z = re.search("(?:_z_|/z)(\\d{1,2})(?:[/_.-]|$)", p.as_posix(), re.IGNORECASE)
    if not m or not z:
        raise ValueError(f"BBBC006 filename or containing z-plane folder unrecognized: {path}")
    d = m.groupdict()
    well = d["well"].lower()
    site = int(d["site"])
    channel = int(d["channel"])
    plane = int(z.group(1))
    if not 0 <= plane <= 33:
        raise ValueError("BBBC006 plane outside published download range")
    return {
        "specimen_id": f"bbbc006_{well}_s{site}_w{channel}",
        "group_id": f"bbbc006_{well}",
        "condition": f"z{plane:02d}",
        "well": well,
        "site": site,
        "channel": channel,
    }


def read_counts(path: str | Path) -> dict[tuple[str, int], float]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = {"Image_Count_Nuclei", "Image_Metadata_Site", "Image_Metadata_Well"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("unrecognized official BBBC006 count columns")
        result = {}
        for row in reader:
            key = (row["Image_Metadata_Well"].lower(), int(row["Image_Metadata_Site"]))
            value = finite("count", row["Image_Count_Nuclei"], 0)
            if key in result:
                raise ValueError("duplicate well/site count")
            result[key] = value
    return result


def index_dataset(
    root: str | Path,
    dataset: str,
    out: str | Path,
    *,
    counts_csv: str | Path | None = None,
    channel: int = 1,
    pairing_confirmed: bool = False,
) -> dict:
    root = Path(root).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing overwrite {out}")
    if channel not in [1, 2]:
        raise ValueError("channel must be 1 or 2")
    if dataset not in ["bbbc005", "bbbc006"]:
        raise ValueError("unknown dataset")
    counts = read_counts(counts_csv) if dataset == "bbbc006" and counts_csv is not None else None
    if dataset == "bbbc006" and counts is None:
        raise ValueError("BBBC006 requires official count CSV")
    rows = []
    errors = []
    seen = set()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff"})
    for p in paths:
        try:
            d = parse_bbbc005(p) if dataset == "bbbc005" else parse_bbbc006(p)
            if d["channel"] != channel:
                continue
            if dataset == "bbbc006":
                d["task_value"] = counts[d["well"], d["site"]]
            key = (d["specimen_id"], d["condition"])
            if key in seen:
                raise ValueError("duplicate specimen/condition; do not merge duplicate downloads")
            seen.add(key)
            rows.append(
                {
                    **d,
                    "path": str(p),
                    "sha256": digest(p),
                    "pairing_verified": int(pairing_confirmed or dataset == "bbbc006"),
                    "label_source": "known simulated count"
                    if dataset == "bbbc005"
                    else "CellProfiler z16 algorithm-generated count",
                    "reference_label": "",
                    "reference_source": "",
                }
            )
        except (ValueError, KeyError) as exc:
            errors.append({"path": str(p), "error": str(exc)})
    if errors:
        atomic_json(out.with_suffix(".index-errors.json"), errors)
        raise ValueError(
            f"{len(errors)} files could not be indexed; see index-errors.json (none silently omitted)"
        )
    if not rows:
        raise ValueError("no matching images")
    write_csv(out, rows)
    report = {
        "dataset": dataset,
        "images": len(rows),
        "specimens": len({r["specimen_id"] for r in rows}),
        "groups": len({r["group_id"] for r in rows}),
        "conditions": sorted({r["condition"] for r in rows}),
        "manifest_sha256": digest(out),
        "pairing_verified": all(r["pairing_verified"] for r in rows),
        "label_source": rows[0]["label_source"],
        "count_source_sha256": digest(counts_csv) if counts_csv else None,
    }
    atomic_json(out.with_suffix(".audit.json"), report)
    return report


def read_image(path: str | Path, size: int) -> np.ndarray:
    integer("size", size, 16)
    import tifffile

    try:
        image = tifffile.imread(path)
    except ValueError as exc:
        if "imagecodecs" in str(exc).lower():
            raise RuntimeError(
                "LZW TIFF needs imagecodecs; install the research-data requirements"
            ) from exc
        raise
    x = as_float(image)
    if x.shape != (size, size):
        x = ndimage.zoom(x, (size / x.shape[0], size / x.shape[1]), order=1, prefilter=False)
    if x.shape != (size, size):
        raise ValueError("unexpected resize output")
    return np.clip(x, 0, 1).astype(np.float32)


def build_panels(
    manifest: str | Path,
    out: str | Path,
    *,
    baseline: str,
    actions: list[str],
    second_baseline: str | None = None,
    count_threshold: float = 50.0,
    size: int = 64,
    seed: int = 42,
    train_correlation: float = 0.95,
    other_correlation: float = 0.5,
    allow_incomplete: bool = False,
    verify_hashes: bool = True,
) -> dict:
    """Group split FIRST; then optional baseline-label correlation within each split.

    Candidate conditions must be disjoint from baseline conditions. Static replay
    cannot pretend that revealing the SAME stored baseline is a fresh repeat.
    """
    finite("count_threshold", count_threshold, 0)
    finite("train_correlation", train_correlation, 0, 1)
    finite("other_correlation", other_correlation, 0, 1)
    integer("size", size, 16)
    if (
        not actions
        or len(set(actions)) != len(actions)
        or baseline in actions
        or (second_baseline in actions)
    ):
        raise ValueError("distinct candidate actions must not include either baseline")
    if second_baseline == baseline:
        raise ValueError("second baseline must differ")
    manifest = Path(manifest).resolve()
    out = Path(out)
    if out.exists():
        raise FileExistsError("output directory exists; choose a fresh path")
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {
        "specimen_id",
        "group_id",
        "condition",
        "path",
        "task_value",
        "pairing_verified",
        "label_source",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest missing required columns")
    groups = defaultdict(dict)
    for r in rows:
        if r["condition"] in groups[r["specimen_id"]]:
            raise ValueError("duplicate manifest condition")
        groups[r["specimen_id"]][r["condition"]] = r
    needed = set([baseline, *actions] + ([second_baseline] if second_baseline else []))
    complete = []
    excluded = []
    for specimen, views in sorted(groups.items()):
        if not needed.issubset(views):
            excluded.append(specimen)
            continue
        selected = [views[c] for c in needed]
        if any(r["pairing_verified"] not in ["1", "true", "True"] for r in selected):
            raise ValueError("same-specimen pairing is not confirmed")
        for key in [
            "group_id",
            "task_value",
            "label_source",
            "reference_label",
            "reference_source",
        ]:
            if len({r.get(key, "") for r in selected}) != 1:
                raise ValueError(f"inconsistent {key} within specimen")
        complete.append(specimen)
    if excluded and (not allow_incomplete):
        raise ValueError(
            f"{len(excluded)} incomplete panels; use --allow-incomplete to explicitly exclude and log"
        )
    if not complete:
        raise ValueError("no complete panels")
    group_ids = [groups[s][baseline]["group_id"] for s in complete]
    split_indices = indices(group_ids, seed)
    result = {}
    out.mkdir(parents=True)
    for split, ix in split_indices.items():
        images = []
        labels = []
        ids = []
        gids = []
        refs = []
        sources = set()
        chosen = []
        for i in ix:
            specimen = complete[i]
            views = groups[specimen]
            first = views[baseline]
            label = int(float(first["task_value"]) >= count_threshold)
            initial = baseline
            if second_baseline:
                corr = train_correlation if split == "train" else other_correlation
                aligned = (
                    np.random.default_rng(seed_for(seed, specimen, "baseline")).random() < corr
                )
                use_second = label if aligned else 1 - label
                initial = second_baseline if use_second else baseline
            sample = []
            for condition in [initial, *actions]:
                row = views[condition]
                path = Path(row["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                if verify_hashes and row.get("sha256") and (digest(path) != row["sha256"]):
                    raise ValueError(f"image hash changed: {path}")
                sample.append(read_image(path, size))
            images.append(sample)
            labels.append(label)
            ids.append(specimen)
            gids.append(first["group_id"])
            refs.append(first.get("reference_label", ""))
            sources.add(first["label_source"])
            chosen.append(initial)
        if any(x != "" for x in refs) and any(x == "" for x in refs):
            raise ValueError("partial independent annotations")
        provenance = {
            "source": "BBBC/generic real-image manifest",
            "manifest_sha256": digest(manifest),
            "pairing_verified": True,
            "label_source": "; ".join(sorted(sources)),
            "count_threshold": count_threshold,
            "interpolation": "linear resize to square, no per-image intensity normalization",
            "size": size,
            "baseline_conditions": chosen,
            "dose_unit": "unit-exposure proxy, not calibrated photons",
            "time_unit": "unit acquisition proxy, not measured seconds",
            "physical_validation": False,
        }
        if refs and refs[0] != "":
            provenance["reference_source"] = groups[ids[0]][baseline].get("reference_source", "")
        panel = Panel(
            np.asarray(images, dtype=np.float32),
            np.asarray(labels),
            ids,
            gids,
            tuple(Intervention(a, 1.0, 1.0) for a in actions),
            provenance,
            np.asarray(refs, dtype=int) if refs and refs[0] != "" else None,
        )
        result[split] = panel
        save_panel(panel, out / f"{split}.npz")
    assert_disjoint(result)
    report = {
        "splits": record(result),
        "excluded_incomplete": excluded,
        "manifest_sha256": digest(manifest),
        "panel_files": {n: str((out / f"{n}.npz").resolve()) for n in result},
    }
    atomic_json(out / "dataset.json", report)
    return report
