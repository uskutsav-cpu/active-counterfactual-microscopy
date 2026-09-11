import csv

import pytest

from counterfactual_microscopy.research.stacks import (
    action_catalog,
    action_name,
    index_lsfm,
    plane_offset_um,
)


def make_stack(root, specimens=("aaa_bbb_ccc", "ddd_eee_fff")):
    for plane in [1, 2, 3]:
        folder = root / str(plane)
        folder.mkdir(parents=True, exist_ok=True)

        for specimen in specimens:
            path = folder / f"{specimen}_{plane:02d}.tiff"
            path.write_bytes(f"{specimen}-{plane}".encode())


def test_plane_offset_and_action_name():
    assert (
        plane_offset_um(
            26,
            focus_plane=26,
            step_um=2.0,
        )
        == 0.0
    )

    assert (
        plane_offset_um(
            8,
            focus_plane=26,
            step_um=2.0,
        )
        == -36.0
    )

    assert (
        plane_offset_um(
            44,
            focus_plane=26,
            step_um=2.0,
        )
        == 36.0
    )

    assert action_name(-36.0) == "focus_m36um"
    assert action_name(0.0) == "focus_0um"
    assert action_name(36.0) == "focus_p36um"


def test_index_lsfm_and_build_action_catalog(tmp_path):
    root = tmp_path / "test"
    make_stack(root)

    manifest = tmp_path / "manifest.csv"

    report = index_lsfm(
        root,
        manifest,
        focus_plane=2,
        step_um=2.0,
        expected_planes=3,
    )

    assert report["images"] == 6
    assert report["specimens"] == 2
    assert report["complete_stacks"] == 2
    assert report["incomplete_stacks"] == 0
    assert report["offset_min_um"] == -2.0
    assert report["offset_max_um"] == 2.0

    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    focus_rows = [row for row in rows if int(row["plane"]) == 2]

    assert len(focus_rows) == 2
    assert all(float(row["offset_um"]) == 0.0 for row in focus_rows)
    assert all(row["is_reference_focus"] == "1" for row in focus_rows)
    assert all(row["biological_label_available"] == "0" for row in rows)

    catalog_path = tmp_path / "actions.json"

    catalog = action_catalog(
        manifest,
        catalog_path,
        planes=[1, 2, 3],
    )

    assert catalog["action_count"] == 3
    assert [action["name"] for action in catalog["actions"]] == [
        "focus_m2um",
        "focus_0um",
        "focus_p2um",
    ]


def test_index_lsfm_rejects_incomplete_stack(tmp_path):
    root = tmp_path / "test"
    make_stack(root)

    (root / "3" / "ddd_eee_fff_03.tiff").unlink()

    with pytest.raises(
        ValueError,
        match="incomplete LSFM stacks",
    ):
        index_lsfm(
            root,
            tmp_path / "manifest.csv",
            focus_plane=2,
            step_um=2.0,
            expected_planes=3,
        )


def test_index_lsfm_rejects_folder_suffix_disagreement(tmp_path):
    root = tmp_path / "test"
    make_stack(root, specimens=("aaa_bbb_ccc",))

    bad = root / "3" / "aaa_bbb_ccc_03.tiff"
    bad.rename(root / "3" / "aaa_bbb_ccc_02.tiff")

    with pytest.raises(
        ValueError,
        match="failed indexing",
    ):
        index_lsfm(
            root,
            tmp_path / "manifest.csv",
            focus_plane=2,
            step_um=2.0,
            expected_planes=3,
        )
