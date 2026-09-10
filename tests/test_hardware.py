import pytest

from counterfactual_microscopy.actions import AcquisitionAction
from counterfactual_microscopy.hardware.mock import MockMicroscopeAdapter


def test_mock_adapter_is_allowlisted() -> None:
    adapter = MockMicroscopeAdapter({"refocus"})
    obs = adapter.acquire(AcquisitionAction("refocus", parameters={"z_um": 0.5}))
    assert obs.metadata["mode"] == "dry-run"

    with pytest.raises(ValueError):
        adapter.acquire(AcquisitionAction("laser_max"))
