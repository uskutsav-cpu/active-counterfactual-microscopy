from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AcquisitionAction:
    """A bounded optical intervention considered for same-specimen reacquisition."""

    name: str
    dose_cost: float = 0.0
    time_cost: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name must be non-empty")
        if self.dose_cost < 0 or self.time_cost < 0:
            raise ValueError("action costs must be non-negative")
