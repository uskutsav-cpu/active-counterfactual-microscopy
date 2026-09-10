"""Opt-in Micro-Manager Java-backend adapter and a fully offline fake core.

This is a software safety boundary, NOT a validated interlock. Vendor/laser hardware
must supply independent interlocks and a supervised operating protocol. MMCore
timeouts do not guarantee that a blocked driver call can be interrupted by Python.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .io import atomic_json
from .types import Intervention
from .validation import finite, integer


class HardwareFault(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyLimits:
    calibration_id: str
    camera_name: str
    min_exposure_ms: float
    max_exposure_ms: float
    power_upper_mw: float
    max_total_dose_mj: float
    max_acquisitions: int
    min_z_um: float
    max_z_um: float
    max_z_offset_um: float
    timeout_ms: int = 3000
    gain_device: str | None = None
    gain_property: str | None = None
    min_gain: float = 0.0
    max_gain: float = 1.0
    detector_max: float = 65535.0

    def __post_init__(self):
        if not self.calibration_id.strip() or "REQUIRED" in self.calibration_id.upper():
            raise ValueError("an actual calibration identifier is required")
        if not self.camera_name.strip():
            raise ValueError("camera_name required")
        finite("min_exposure_ms", self.min_exposure_ms, 1e-06)
        finite("max_exposure_ms", self.max_exposure_ms, self.min_exposure_ms)
        finite("power_upper_mw", self.power_upper_mw, 1e-12)
        finite("max_total_dose_mj", self.max_total_dose_mj, 1e-12)
        finite("min_z_um", self.min_z_um)
        finite("max_z_um", self.max_z_um, self.min_z_um)
        finite("max_z_offset_um", self.max_z_offset_um, 0)
        finite("min_gain", self.min_gain, 0)
        finite("max_gain", self.max_gain, self.min_gain)
        finite("detector_max", self.detector_max, 1)
        integer("max_acquisitions", self.max_acquisitions, 1)
        integer("timeout_ms", self.timeout_ms, 1)
        if bool(self.gain_device) != bool(self.gain_property):
            raise ValueError("gain device and property must be specified together")


class PycroAdapter:
    """Explicitly injected Java-backend Core, closed shutter on completion/fault.

    Methods use snake_case Java bridge names as documented by Pycro-Manager.
    This adapter has been mock-tested, not tested on a microscope. Python-backend
    Core has API differences and is intentionally not silently supported.
    """

    def __init__(self, core, limits: SafetyLimits, *, armed: bool = False):
        self.core = core
        self.limits = limits
        self.armed = armed
        self.faulted = False
        self.total_dose_mj = 0.0
        self.acquisitions = 0
        self.log = []
        self.lock = threading.Lock()
        self.origin_z = None

    @classmethod
    def connect(cls, limits: SafetyLimits, *, armed: bool = False):
        if not armed:
            raise ValueError("connection refused without explicit arming")
        from pycromanager import Core

        return cls(Core(), limits, armed=True)

    def validate(self, parameters: dict) -> dict:
        if set(parameters) - {"exposure_ms", "z_offset_um", "gain"}:
            raise ValueError("unsupported device command")
        if "exposure_ms" not in parameters:
            raise ValueError("each exposure needs an explicit exposure_ms")
        exposure = finite(
            "exposure_ms",
            parameters["exposure_ms"],
            self.limits.min_exposure_ms,
            self.limits.max_exposure_ms,
        )
        offset = finite(
            "z_offset_um",
            parameters.get("z_offset_um", 0.0),
            -self.limits.max_z_offset_um,
            self.limits.max_z_offset_um,
        )
        gain = parameters.get("gain")
        if gain is not None:
            if not self.limits.gain_device:
                raise ValueError("gain control not calibrated/allowed")
            gain = finite("gain", gain, self.limits.min_gain, self.limits.max_gain)
        dose = self.limits.power_upper_mw * exposure / 1000.0
        if self.total_dose_mj + dose > self.limits.max_total_dose_mj + 1e-12:
            raise ValueError("cumulative optical-energy budget exceeded")
        if self.acquisitions >= self.limits.max_acquisitions:
            raise ValueError("hardware acquisition count exceeded")
        return {"exposure_ms": exposure, "z_offset_um": offset, "gain": gain, "dose_mj_upper": dose}

    def acquire(self, action: Intervention) -> np.ndarray:
        if not self.armed or self.faulted:
            raise HardwareFault("adapter is disarmed or fault-latched")
        if not self.lock.acquire(blocking=False):
            raise HardwareFault("concurrent acquisition forbidden")
        snapshot = {}
        result = None
        event = {"action": action.name, "calibration_id": self.limits.calibration_id}
        started = time.perf_counter()
        primary_error = None
        restore_errors = []
        try:
            settings = self.validate(action.parameters)
            event["requested"] = settings
            if str(self.core.get_camera_device()) != self.limits.camera_name:
                raise HardwareFault("camera differs from calibration")
            self.core.set_shutter_open(False)
            self.core.set_timeout_ms(self.limits.timeout_ms)
            snapshot["exposure"] = float(self.core.get_exposure())
            snapshot["auto_shutter"] = bool(self.core.get_auto_shutter())
            snapshot["z"] = float(self.core.get_position())
            if self.origin_z is None:
                self.origin_z = snapshot["z"]
            target_z = self.origin_z + settings["z_offset_um"]
            finite("target_z", target_z, self.limits.min_z_um, self.limits.max_z_um)
            if settings["gain"] is not None:
                snapshot["gain"] = self.core.get_property(
                    self.limits.gain_device, self.limits.gain_property
                )
            self.core.set_auto_shutter(True)
            self.core.set_exposure(settings["exposure_ms"])
            self.core.set_position(target_z)
            self.core.wait_for_device(self.core.get_focus_device())
            if settings["gain"] is not None:
                self.core.set_property(
                    self.limits.gain_device, self.limits.gain_property, str(settings["gain"])
                )
            actual_exposure = float(self.core.get_exposure())
            actual_z = float(self.core.get_position())
            if abs(actual_exposure - settings["exposure_ms"]) > max(
                0.001, 0.001 * settings["exposure_ms"]
            ):
                raise HardwareFault("exposure readback mismatch")
            if abs(actual_z - target_z) > 0.05:
                raise HardwareFault("focus readback mismatch")
            self.total_dose_mj += settings["dose_mj_upper"]
            self.acquisitions += 1
            self.core.snap_image()
            tagged = self.core.get_tagged_image()
            image = np.asarray(tagged.pix).reshape(
                int(tagged.tags["Height"]), int(tagged.tags["Width"])
            )
            if not np.isfinite(image).all() or image.min() < 0:
                raise HardwareFault("invalid detector pixels")
            result = np.clip(image.astype(np.float32) / self.limits.detector_max, 0, 1)
            event.update(
                actual_exposure_ms=actual_exposure,
                actual_z_um=actual_z,
                pixel_shape=list(result.shape),
            )
        except BaseException as exc:
            primary_error = exc
            self.faulted = True
            event["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            operations = [("shutter", lambda: self.core.set_shutter_open(False))]
            if "exposure" in snapshot:
                operations.append(
                    ("exposure", lambda: self.core.set_exposure(snapshot["exposure"]))
                )
            if "z" in snapshot:
                operations.append(("focus", lambda: self.core.set_position(snapshot["z"])))
            if "gain" in snapshot:
                operations.append(
                    (
                        "gain",
                        lambda: self.core.set_property(
                            self.limits.gain_device, self.limits.gain_property, snapshot["gain"]
                        ),
                    )
                )
            if "auto_shutter" in snapshot:
                operations.append(
                    ("auto_shutter", lambda: self.core.set_auto_shutter(snapshot["auto_shutter"]))
                )
            for name, operation in operations:
                try:
                    operation()
                except Exception as exc:
                    restore_errors.append(f"{name}: {exc}")
                    self.faulted = True
            event.update(
                wall_seconds=time.perf_counter() - started,
                cumulative_dose_mj_upper=self.total_dose_mj,
                restore_errors=restore_errors,
                faulted=self.faulted,
            )
            self.log.append(event)
            self.lock.release()
        if primary_error is not None:
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error
            raise HardwareFault(str(primary_error)) from primary_error
        if restore_errors:
            raise HardwareFault("failed safe-state restoration: " + "; ".join(restore_errors))
        return result

    def disarm(self) -> None:
        self.armed = False
        try:
            self.core.set_shutter_open(False)
        except Exception as exc:
            self.faulted = True
            raise HardwareFault("could not close shutter") from exc


class LiveProvider:
    """Label-free provider for the same inference loop; requires physically calibrated data."""

    def __init__(
        self,
        adapter: PycroAdapter,
        baseline_action: Intervention,
        actions: tuple[Intervention, ...],
    ):
        self.adapter = adapter
        self.baseline_action = baseline_action
        self.actions = actions
        self.used = set()
        self.initial = None

    def baseline(self) -> np.ndarray:
        if self.initial is None:
            self.initial = self.adapter.acquire(self.baseline_action)
        return self.initial.copy()

    def acquire(self, index: int) -> np.ndarray:
        integer("index", index)
        if index >= len(self.actions) or index in self.used:
            raise ValueError("invalid or repeated live action")
        self.used.add(index)
        return self.adapter.acquire(self.actions[index])


class FakeCore:
    """Offline test double. Deliberately cannot connect to a microscope."""

    def __init__(self, fail_snap: bool = False):
        self.exposure = 10.0
        self.z = 0.0
        self.auto = True
        self.shutter = False
        self.fail_snap = fail_snap
        self.calls = []
        self.properties = {}
        self.timeout = 0

    def get_camera_device(self):
        return "FAKE_CAMERA"

    def get_focus_device(self):
        return "FAKE_FOCUS"

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, x):
        self.exposure = float(x)
        self.calls.append(("exposure", x))

    def get_auto_shutter(self):
        return self.auto

    def set_auto_shutter(self, x):
        self.auto = bool(x)

    def set_shutter_open(self, x):
        self.shutter = bool(x)
        self.calls.append(("shutter", x))

    def set_timeout_ms(self, x):
        self.timeout = x

    def get_position(self):
        return self.z

    def set_position(self, x):
        self.z = float(x)
        self.calls.append(("z", x))

    def wait_for_device(self, name):
        self.calls.append(("wait", name))

    def get_property(self, device, name):
        return self.properties.get((device, name), "1")

    def set_property(self, device, name, value):
        self.properties[device, name] = value

    def snap_image(self):
        self.calls.append(("snap", self.exposure))
        if self.fail_snap:
            raise RuntimeError("simulated camera failure")

    def get_tagged_image(self):
        yy, xx = np.mgrid[:32, :32]
        pixels = np.clip(
            (1000 + 15000 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 30)) * self.exposure / 10,
            0,
            65535,
        ).astype(np.uint16)
        return SimpleNamespace(pix=pixels.ravel(), tags={"Height": 32, "Width": 32})


def fake_limits() -> SafetyLimits:
    return SafetyLimits(
        "OFFLINE_FAKE_ONLY", "FAKE_CAMERA", 1, 30, 1, 0.2, 5, -5, 5, 2, detector_max=65535
    )


def dry_run(out: str | Path) -> dict:
    adapter = PycroAdapter(FakeCore(), fake_limits(), armed=True)
    image = adapter.acquire(Intervention("demo", 1, 0.1, {"exposure_ms": 10.0, "z_offset_um": 1.0}))
    adapter.disarm()
    result = {
        "mode": "offline_fake_core",
        "real_hardware_used": False,
        "image_shape": list(image.shape),
        "log": adapter.log,
    }
    atomic_json(out, result)
    return result
