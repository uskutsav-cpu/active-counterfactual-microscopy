import csv
import json
import stat
import zipfile

import numpy as np
import pytest
import tifffile

from counterfactual_microscopy.research.datasets import (
    build_panels,
    index_dataset,
    parse_bbbc005,
    parse_bbbc006,
    read_counts,
)
from counterfactual_microscopy.research.downloads import plan, safe_extract
from counterfactual_microscopy.research.io import (
    atomic_json,
    digest,
    load_panel,
    save_panel,
    seed_for,
)
from counterfactual_microscopy.research.splits import assert_disjoint, indices
from counterfactual_microscopy.research.statistics import metrics, paired_bootstrap, risk_coverage


def row(i, target, verdict, p):
    return {
        "specimen_id": str(i),
        "group_id": str(i // 2),
        "target": target,
        "verdict": verdict,
        "posterior": p,
        "dose": 1.0,
        "acquisition_seconds": 1.0,
        "eligible": True,
    }


def test_metric_denominators():
    rows = [
        row(0, 0, "supported", 0.9),
        row(1, 0, "abstain", 0.5),
        row(2, 1, "supported", 0.9),
        row(3, 1, "falsified", 0.1),
    ]
    m = metrics(rows)
    assert m["false_support_conditional"] == 0.5
    assert m["false_support_joint"] == 0.25
    assert m["supported_prediction_risk"] == 0.5
    assert m["selective_verdict_risk"] == pytest.approx(2 / 3)
    assert m["coverage"] == 0.75


def test_undefined_metrics_not_zero():
    m = metrics([row(0, 1, "abstain", 0.5)])
    assert m["supported_prediction_risk"] is None and m["false_support_conditional"] is None
    assert m["auroc"] is None and m["selective_verdict_risk"] is None


def test_bootstrap_pairing_and_zero_difference():
    rows = [row(i, i % 2, "supported", 0.8) for i in range(20)]
    result = paired_bootstrap(rows, rows, "false_support_conditional", 20, 1)
    assert result["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        paired_bootstrap(rows, rows[:-1], "coverage", 20)


def test_risk_curve_ties():
    rows = [row(i, i % 2, "supported", 0.8) for i in range(10)]
    curve = risk_coverage(rows)
    assert len(curve) == 1 and curve[0]["coverage"] == 1


def test_panel_roundtrip_and_overwrite(tmp_path, research_system):
    train, *_ = research_system
    p = train.subset([0, 1, 2])
    path = tmp_path / "panel.npz"
    save_panel(p, path)
    other = load_panel(path)
    np.testing.assert_array_equal(p.images, other.images)
    assert p.actions == other.actions
    with pytest.raises(FileExistsError):
        save_panel(p, path)


def test_json_and_stable_seed(tmp_path):
    p = tmp_path / "x.json"
    atomic_json(p, {"x": float("nan"), "y": np.int64(2)})
    assert json.loads(p.read_text()) == {"x": None, "y": 2}
    assert seed_for("a", 1) == seed_for("a", 1)
    assert seed_for("a", 1) != seed_for("a", 2)
    assert len(digest(p)) == 64


def test_group_separation(research_system):
    train, cal, *_ = research_system
    assert_disjoint({"train": train, "cal": cal})
    with pytest.raises(ValueError):
        assert_disjoint({"train": train, "cal": train})
    split = indices([f"g{i // 2}" for i in range(40)])
    seen = set()
    for ix in split.values():
        g = {int(i) // 2 for i in ix}
        assert not seen.intersection(g)
        seen.update(g)


@pytest.mark.parametrize("name", ["bad.tif", "SIMCEPImages_A01_C999_F1_s01_w1.TIF"])
def test_bad_bbbc005_name(name):
    with pytest.raises(ValueError):
        parse_bbbc005(name)


def test_bbbc_parsers():
    d = parse_bbbc005("SIMCEPImages_A01_C15_F3_s02_w2.TIF")
    assert d["task_value"] == 15 and d["condition"] == "f03"
    d = parse_bbbc006("/data/BBBC006_v1_images_z_16/mcf-z-stacks-03212011_a02_s1_w1abcdef.tif")
    assert d["condition"] == "z16" and d["group_id"] == "bbbc006_a02"
    with pytest.raises(ValueError):
        parse_bbbc006("/no-plane/mcf_a02_s1_w1abcdef.tif")


def test_counts_official_columns(tmp_path):
    p = tmp_path / "counts.csv"
    p.write_text("Image_Count_Nuclei,Image_Metadata_Site,Image_Metadata_Well\n68,1,a02\n")
    assert read_counts(p) == {("a02", 1): 68.0}


def test_download_plan_no_network():
    assert len(plan("bbbc006", [8, 16, 24])) == 4
    assert plan("bbbc005")[0].url.startswith("https://data.broadinstitute.org/")
    with pytest.raises(ValueError):
        plan("bbbc006", [16, 16])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/windows", "a\\b"])
def test_archive_traversal_rejected(tmp_path, name):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, "bad")
    with pytest.raises(ValueError):
        safe_extract(p, tmp_path / "out")


def test_archive_symlink_and_size(tmp_path):
    p = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 511) << 16
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(info, "target")
    with pytest.raises(ValueError):
        safe_extract(p, tmp_path / "out")
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x", "a" * 100)
    with pytest.raises(ValueError):
        safe_extract(p, tmp_path / "out", max_uncompressed=50)


def test_archive_valid_and_no_overwrite(tmp_path):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("folder/a.txt", "hello")
    out = safe_extract(p, tmp_path / "out")
    assert out[0].read_text() == "hello"
    with pytest.raises(FileExistsError):
        safe_extract(p, tmp_path / "out")


def test_real_format_fixture_ingestion(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    records = []
    for well in range(1, 21):
        for site in [1, 2]:
            records.append(
                {
                    "Image_Count_Nuclei": 10 + well * 4,
                    "Image_Metadata_Site": site,
                    "Image_Metadata_Well": f"a{well:02d}",
                }
            )
            for z in [8, 16, 24]:
                directory = root / f"BBBC006_v1_images_z_{z:02d}"
                directory.mkdir(exist_ok=True)
                image = np.random.default_rng(well * 100 + site).integers(
                    0, 20000, (20, 20), dtype=np.uint16
                )
                tifffile.imwrite(directory / f"mcf_a{well:02d}_s{site}_w1uuid.tif", image)
    counts = tmp_path / "counts.csv"
    with counts.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = tmp_path / "manifest.csv"
    result = index_dataset(root, "bbbc006", manifest, counts_csv=counts)
    assert result["images"] == 120 and result["groups"] == 20
    built = build_panels(
        manifest, tmp_path / "panels", baseline="z08", actions=["z16", "z24"], size=24
    )
    panels = {n: load_panel(p) for n, p in built["panel_files"].items()}
    assert_disjoint(panels)
    assert sum(len(p.labels) for p in panels.values()) == 40
    assert "algorithm" in panels["train"].provenance["label_source"]
