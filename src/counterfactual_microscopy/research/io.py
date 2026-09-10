"""Atomic reports, non-pickle panel files, hashes, and reproducibility metadata."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .types import Intervention, Panel


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return str(value) if isinstance(value, Path) else value


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(safe_json(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(*items: object) -> int:
    return int.from_bytes(
        hashlib.sha256(json.dumps(items, sort_keys=True).encode()).digest()[:4], "little"
    )


def source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment() -> dict:
    versions = {}
    for name in [
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "PyYAML",
        "matplotlib",
        "tifffile",
        "imagecodecs",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True, timeout=3
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "research_source_sha256": source_digest(),
    }


def save_panel(panel: Panel, path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "actions": [asdict(a) for a in panel.actions],
        "provenance": panel.provenance,
    }
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            images=panel.images,
            labels=panel.labels,
            specimen_ids=np.asarray(panel.specimen_ids),
            group_ids=np.asarray(panel.group_ids),
            reference=np.array([], dtype=int) if panel.reference is None else panel.reference,
            metadata=np.asarray(json.dumps(safe_json(metadata))),
        )


def load_panel(path: str | Path) -> Panel:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta.get("schema") != 1:
            raise ValueError("unsupported panel schema")
        ref = data["reference"].copy()
        return Panel(
            data["images"].copy(),
            data["labels"].copy(),
            data["specimen_ids"].tolist(),
            data["group_ids"].tolist(),
            tuple(Intervention(**a) for a in meta["actions"]),
            meta["provenance"],
            ref if len(ref) else None,
        )


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty results table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(safe_json(v)) if isinstance(v, (dict, list)) else safe_json(v)
                    for k, v in row.items()
                }
            )
