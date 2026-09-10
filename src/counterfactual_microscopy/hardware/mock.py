from dataclasses import dataclass

from ..actions import AcquisitionAction
from .base import AcquisitionCost, HardwareObservation


@dataclass(slots=True)
class MockMicroscopeAdapter:
    allowed_actions: set[str]

    def validate_action(self, action: AcquisitionAction) -> None:
        if action.name not in self.allowed_actions:
            raise ValueError(f"action {action.name!r} is not allowlisted")

    def estimate_cost(self, action: AcquisitionAction) -> AcquisitionCost:
        self.validate_action(action)
        return AcquisitionCost(dose=action.dose_cost, seconds=action.time_cost)

    def acquire(self, action: AcquisitionAction) -> HardwareObservation:
        self.validate_action(action)
        return HardwareObservation(
            action_name=action.name,
            payload_uri=None,
            metadata={"mode": "dry-run", "parameters": action.parameters},
        )
