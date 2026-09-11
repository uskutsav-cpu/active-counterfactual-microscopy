"""Executable workflows. No pseudocode commands and no automatic live-hardware connection."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import load, smoke
from .io import load_panel, safe_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfm-lab", description="Active microscopy computational research engine"
    )
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("smoke", help="small all-policy image experiment; no downloads")
    q.add_argument("--out", required=True)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("run", help="run a validated YAML configuration")
    q.add_argument("--config", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("sweep", help="run a prespecified grid, logging failures and successes")
    q.add_argument("--config", required=True)
    q.add_argument("--grid", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("report", help="rebuild report and figures from frozen test rows")
    q.add_argument("--run", required=True)
    q = sub.add_parser("validate-panel")
    q.add_argument("path")
    q = sub.add_parser("dataset-plan", help="show sizes/URLs; does not download")
    q.add_argument("dataset", choices=["bbbc005", "bbbc006"])
    q.add_argument("--planes", nargs="+", type=int)
    q = sub.add_parser("dataset-download", help="opt-in bounded downloads from official BBBC")
    q.add_argument("dataset", choices=["bbbc005", "bbbc006"])
    q.add_argument("--planes", nargs="+", type=int)
    q.add_argument("--out", required=True)
    q.add_argument("--max-gb", required=True, type=float)
    q.add_argument("--accept-download", action="store_true")
    q.add_argument("--extract", action="store_true")
    q = sub.add_parser("dataset-index", help="create audited image metadata manifest")
    q.add_argument("dataset", choices=["bbbc005", "bbbc006"])
    q.add_argument("--root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--counts-csv")
    q.add_argument("--channel", type=int, default=1)
    q.add_argument("--confirm-pairing", action="store_true")
    q = sub.add_parser("dataset-build", help="build group-disjoint replay panels")
    q.add_argument("--manifest", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--baseline", required=True)
    q.add_argument("--actions", nargs="+", required=True)
    q.add_argument("--second-baseline")
    q.add_argument("--count-threshold", type=float, default=50.0)
    q.add_argument("--size", type=int, default=64)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--train-correlation", type=float, default=0.95)
    q.add_argument("--other-correlation", type=float, default=0.5)
    q.add_argument("--allow-incomplete", action="store_true")
    q = sub.add_parser(
        "stack-index",
        help="create an audited same-specimen LSFM optical-stack manifest",
    )
    q.add_argument("--root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--focus-plane", type=int, default=26)
    q.add_argument("--step-um", type=float, default=2.0)
    q.add_argument("--expected-planes", type=int, default=51)
    q = sub.add_parser(
        "stack-actions",
        help="freeze a multi-plane optical-action catalog",
    )
    q.add_argument("--manifest", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--planes", nargs="+", type=int)
    q = sub.add_parser("hardware-dry-run", help="offline fake camera only")
    q.add_argument("--out", required=True)
    q = sub.add_parser("verify-run", help="check hashes of frozen result artifacts")
    q.add_argument("--run", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ["smoke", "run"]:
            from .experiment import run

            result = run(
                smoke() if args.command == "smoke" else load(args.config), args.out, args.resume
            )
        elif args.command == "sweep":
            from .experiment import sweep

            grid = yaml.safe_load(Path(args.grid).read_text())
            if not isinstance(grid, dict):
                raise ValueError("grid must be a mapping")
            result = sweep(load(args.config), grid, args.out, args.resume)
        elif args.command == "report":
            from .reporting import render_report

            result = render_report(args.run)
        elif args.command == "validate-panel":
            panel = load_panel(args.path)
            result = {
                "images": list(panel.images.shape),
                "groups": len(set(panel.group_ids)),
                "pairing_verified": panel.provenance.get("pairing_verified"),
                "provenance": panel.provenance,
                "actions": [asdict(a) for a in panel.actions],
            }
        elif args.command == "dataset-plan":
            from .downloads import plan

            items = plan(args.dataset, args.planes)
            result = {
                "downloads": [asdict(i) for i in items],
                "approximate_gb": sum(i.approximate_bytes for i in items) / 1000000000.0,
                "download_started": False,
            }
        elif args.command == "dataset-download":
            from .downloads import download_dataset

            if not args.accept_download:
                raise ValueError("explicit --accept-download is required")
            result = download_dataset(
                args.dataset, args.out, args.planes, args.max_gb, args.extract
            )
        elif args.command == "dataset-index":
            from .datasets import index_dataset

            result = index_dataset(
                args.root,
                args.dataset,
                args.out,
                counts_csv=args.counts_csv,
                channel=args.channel,
                pairing_confirmed=args.confirm_pairing,
            )
        elif args.command == "dataset-build":
            from .datasets import build_panels

            result = build_panels(
                args.manifest,
                args.out,
                baseline=args.baseline,
                actions=args.actions,
                second_baseline=args.second_baseline,
                count_threshold=args.count_threshold,
                size=args.size,
                seed=args.seed,
                train_correlation=args.train_correlation,
                other_correlation=args.other_correlation,
                allow_incomplete=args.allow_incomplete,
            )
        elif args.command == "stack-index":
            from .stacks import index_lsfm

            result = index_lsfm(
                args.root,
                args.out,
                focus_plane=args.focus_plane,
                step_um=args.step_um,
                expected_planes=args.expected_planes,
            )
        elif args.command == "stack-actions":
            from .stacks import action_catalog

            result = action_catalog(
                args.manifest,
                args.out,
                planes=args.planes,
            )
        elif args.command == "hardware-dry-run":
            from .hardware import dry_run

            result = dry_run(args.out)
        else:
            from .io import digest

            root = Path(args.run)
            hashes = json.loads((root / "artifact_hashes.json").read_text())
            errors = [
                p for p, h in hashes.items() if not (root / p).is_file() or digest(root / p) != h
            ]
            result = {"valid": not errors, "mismatches": errors, "files_checked": len(hashes)}
            if errors:
                print(json.dumps(result, indent=2))
                return 1
        print(json.dumps(safe_json(result), indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get("failed", 0) else 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
