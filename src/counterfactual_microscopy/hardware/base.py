from dataclasses import dataclass
from typing import Protocol

from ..actions import AcquisitionAction


@dataclass(frozen=True, slots=True)
class AcquisitionCost:
    dose: float
    seconds: float


@dataclass(frozen=True, slots=True)
class HardwareObservation:
    action_name: str
    payload_uri: str | None = None
    metadata: dict | None = None


class MicroscopeAdapter(Protocol):
    """Minimal safe boundary between scientific policy and device control."""

    def validate_action(self, action: AcquisitionAction) -> None: ...

    def estimate_cost(self, action: AcquisitionAction) -> AcquisitionCost: ...

    def acquire(self, action: AcquisitionAction) -> HardwareObservation: ...
