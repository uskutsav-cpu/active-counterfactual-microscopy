import json
from dataclasses import replace

import numpy as np
import pytest

from counterfactual_microscopy.research.cli import main
from counterfactual_microscopy.research.config import Config
from counterfactual_microscopy.research.experiment import run
from counterfactual_microscopy.research.hardware import (
    FakeCore,
    HardwareFault,
    PycroAdapter,
    fake_limits,
)
from counterfactual_microscopy.research.types import Intervention


def action(**kwargs):
    return Intervention("acquire", 1, 0.1, {"exposure_ms": 10.0, **kwargs})


def test_hardware_disarmed_no_snap():
    core = FakeCore()
    adapter = PycroAdapter(core, fake_limits())
    with pytest.raises(HardwareFault):
        adapter.acquire(action())
    assert not core.calls


def test_hardware_success_restores_and_closes():
    core = FakeCore()
    adapter = PycroAdapter(core, fake_limits(), armed=True)
    image = adapter.acquire(action(exposure_ms=20.0, z_offset_um=1.0))
    assert image.shape == (32, 32) and np.isfinite(image).all()
    assert core.exposure == 10.0 and core.z == 0.0 and (not core.shutter)
    assert adapter.total_dose_mj == pytest.approx(0.02)


def test_hardware_failed_exposure_charged_and_latched():
    core = FakeCore(fail_snap=True)
    adapter = PycroAdapter(core, fake_limits(), armed=True)
    with pytest.raises(HardwareFault):
        adapter.acquire(action())
    assert adapter.total_dose_mj == 0.01 and adapter.faulted and (not core.shutter)
    with pytest.raises(HardwareFault):
        adapter.acquire(action())


@pytest.mark.parametrize(
    "params",
    [
        {"exposure_ms": 100},
        {"exposure_ms": float("nan")},
        {"z_offset_um": 100},
        {"gain": 2},
        {"unknown": 1},
    ],
)
def test_hardware_invalid_commands_no_exposure(params):
    core = FakeCore()
    adapter = PycroAdapter(core, fake_limits(), armed=True)
    with pytest.raises((ValueError, HardwareFault)):
        adapter.acquire(action(**params))
    assert not any((k == "snap" for k, v in core.calls))


def test_hardware_budget_and_camera():
    core = FakeCore()
    adapter = PycroAdapter(core, replace(fake_limits(), max_total_dose_mj=0.005), armed=True)
    with pytest.raises(HardwareFault):
        adapter.acquire(action())
    assert adapter.acquisitions == 0
    adapter = PycroAdapter(core, replace(fake_limits(), camera_name="OTHER"), armed=True)
    with pytest.raises(HardwareFault):
        adapter.acquire(action())


def test_hardware_restore_failure_latches():

    class BrokenShutter(FakeCore):
        def set_shutter_open(self, value):
            raise RuntimeError("shutter stuck")

    adapter = PycroAdapter(BrokenShutter(), fake_limits(), armed=True)
    with pytest.raises(HardwareFault):
        adapter.acquire(action())
    assert adapter.faulted and adapter.log[-1]["restore_errors"]


def test_hardware_concurrency_rejected():
    adapter = PycroAdapter(FakeCore(), fake_limits(), armed=True)
    adapter.lock.acquire()
    try:
        with pytest.raises(HardwareFault):
            adapter.acquire(action())
    finally:
        adapter.lock.release()


def test_cli_offline(tmp_path):
    assert main(["dataset-plan", "bbbc006", "--planes", "8", "16"]) == 0
    assert (
        main(["dataset-download", "bbbc005", "--out", str(tmp_path / "data"), "--max-gb", "3"]) == 2
    )
    assert main(["hardware-dry-run", "--out", str(tmp_path / "hardware.json")]) == 0
    assert json.loads((tmp_path / "hardware.json").read_text())["real_hardware_used"] is False


def test_end_to_end_resume_and_hashes(tmp_path):
    cfg = Config(
        train_n=100,
        calibration_n=100,
        development_n=20,
        test_n=16,
        image_size=24,
        policies=["eig", "random", "no_reacquisition"],
        budgets=[1],
        n_states=3,
        bootstrap_repeats=10,
        save_models=False,
        seed=10,
    )
    out = tmp_path / "run"
    result = run(cfg, out)
    assert result["state"] == "COMPLETE" and result["episodes"] == 48
    assert (out / "figures/risk_coverage.png").is_file()
    assert main(["verify-run", "--run", str(out)]) == 0
    assert run(cfg, out, resume=True)["reused_complete_run"]
    with pytest.raises(FileExistsError):
        run(cfg, out)
    with pytest.raises(ValueError):
        run(replace(cfg, seed=99), out, resume=True)
    (out / "summary.csv").write_text("tamper")
    assert main(["verify-run", "--run", str(out)]) == 1
