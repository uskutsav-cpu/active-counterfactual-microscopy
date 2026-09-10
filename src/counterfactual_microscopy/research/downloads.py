"""Opt-in official BBBC downloads with byte limits, resume, and safe ZIP extraction.

Local hashes record downloaded bytes; they are not publisher-provided checksums.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .io import atomic_json, digest
from .validation import finite


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    approximate_bytes: int


def plan(dataset: str, planes: list[int] | None = None) -> list[Download]:
    base = "https://data.broadinstitute.org/bbbc"
    if dataset == "bbbc005":
        return [
            Download("BBBC005_v1_images.zip", f"{base}/BBBC005/BBBC005_v1_images.zip", 1900000000)
        ]
    if dataset == "bbbc006":
        planes = planes or [8, 16, 24]
        if len(set(planes)) != len(planes) or any(
            type(z) is not int or not 0 <= z <= 33 for z in planes
        ):
            raise ValueError("BBBC006 planes must be distinct integers in 0..33")
        result = [
            Download(
                f"BBBC006_v1_images_z_{z:02d}.zip",
                f"{base}/BBBC006/BBBC006_v1_images_z_{z:02d}.zip",
                820000000,
            )
            for z in planes
        ]
        return result + [
            Download("BBBC006_v1_counts.csv", f"{base}/BBBC006/BBBC006_v1_counts.csv", 70000)
        ]
    raise ValueError("unknown dataset")


def fetch(item: Download, directory: str | Path, max_bytes: int, timeout: float = 30.0) -> Path:
    """Resume a partial response only when Content-Range agrees. Never disable TLS."""
    finite("timeout", timeout, 1)
    finite("max_bytes", max_bytes, 1)
    if not item.url.startswith("https://data.broadinstitute.org/bbbc/"):
        raise ValueError("download URL outside official allowlist")
    if Path(item.name).name != item.name or item.name in {"", ".", ".."} or "\\" in item.name:
        raise ValueError("unsafe download filename")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / item.name
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists():
        if not receipt.exists():
            raise FileExistsError("existing download has no checksum receipt")
        meta = json.loads(receipt.read_text())
        if meta["url"] != item.url or meta["sha256"] != digest(destination):
            raise ValueError("download receipt mismatch")
        if destination.stat().st_size > max_bytes:
            raise ValueError("existing download exceeds remaining byte budget")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "counterfactual-microscopy-research/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        if offset and status == 206:
            cr = response.headers.get("Content-Range", "")
            if not cr.startswith(f"bytes {offset}-"):
                raise ValueError("server returned wrong resume offset")
            mode = "ab"
        elif status == 200:
            offset = 0
            mode = "wb"
        else:
            raise ValueError(f"unexpected download response {status}")
        announced = response.headers.get("Content-Length")
        if announced and offset + int(announced) > max_bytes:
            raise ValueError("download exceeds byte limit")
        written = offset
        with partial.open(mode) as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                if written + len(block) > max_bytes:
                    raise ValueError("download exceeded byte limit")
                f.write(block)
                written += len(block)
        if announced and written - offset != int(announced):
            raise OSError("truncated download; partial retained")
    if destination.suffix.lower() == ".zip":
        if not zipfile.is_zipfile(partial):
            raise ValueError("response is not a ZIP archive")
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
    elif destination.suffix.lower() == ".csv":
        with partial.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"Image_Count_Nuclei", "Image_Metadata_Site", "Image_Metadata_Well"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("CSV response does not have BBBC006 count columns")
            if next(reader, None) is None:
                raise ValueError("empty counts CSV response")
    else:
        raise ValueError("unsupported dataset file type")
    os.replace(partial, destination)
    atomic_json(
        receipt,
        {
            "url": item.url,
            "bytes": written,
            "sha256": digest(destination),
            "upstream_checksum_supplied": False,
        },
    )
    return destination


def safe_extract(
    archive: str | Path, destination: str | Path, max_uncompressed: int = 12000000000
) -> list[Path]:
    """Reject traversal, symlinks, duplicates, existing-file overwrite and ZIP bombs."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = []
    seen = set()
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > max_uncompressed:
            raise ValueError("uncompressed archive too large")
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in item.filename
                or re.match("^[A-Za-z]:", item.filename)
            ):
                raise ValueError("unsafe archive path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("archive symlinks forbidden")
            target = (root / str(name)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("archive path escape")
            if target in seen:
                raise ValueError("duplicate archive member")
            seen.add(target)
            if not item.is_dir() and target.exists():
                raise FileExistsError(f"refusing overwrite {target}")
            targets.append((item, target))
        for item, target in targets:
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    return [p for i, p in targets if not i.is_dir()]


def download_dataset(
    dataset: str,
    directory: str | Path,
    planes: list[int] | None,
    max_gb: float,
    extract: bool = False,
) -> list[dict]:
    finite("max_gb", max_gb, 0.001)
    limit = int(max_gb * 1000000000)
    items = plan(dataset, planes)
    if sum(i.approximate_bytes for i in items) > limit:
        raise ValueError("planned data exceeds explicit max-gb budget")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < sum(i.approximate_bytes for i in items) * (
        4 if extract else 1.2
    ):
        raise OSError("insufficient disk space for download/extraction estimate")
    out = []
    used = 0
    for item in items:
        p = fetch(item, root / "archives", limit - used)
        used += p.stat().st_size
        extracted = None
        if extract and p.suffix.lower() == ".zip":
            extracted = root / "extracted" / p.stem
            marker = extracted / ".extract-complete.json"
            if marker.exists():
                if json.loads(marker.read_text())["archive_sha256"] != digest(p):
                    raise ValueError("extraction checksum mismatch")
            else:
                safe_extract(p, extracted)
                atomic_json(marker, {"archive_sha256": digest(p)})
        out.append(
            {
                "archive": str(p),
                "extracted": str(extracted) if extracted else None,
                "sha256": digest(p),
            }
        )
    return out
