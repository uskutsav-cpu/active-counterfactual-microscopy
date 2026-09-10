# Physical acquisition gate

**The packaged adapter is mock-tested software, not validated microscope control. The standard CLI has no live-acquisition command.** Its offline hardware-dry-run cannot operate a device.

The adapter follows Pycro-Manager's Java-backend, snake_case Core API. The Python backend may differ and is not silently substituted. A collaborating instrument operator must verify every device method and property before arming it.

## Required instrument calibration

Record the actual camera identity; detector scale; exposure range and readback tolerance; focus origin, safe absolute range and offset range; gain device/property/range when enabled; an upper bound on illumination power at the specimen; and total allowed energy/acquisition count. Assign a real calibration ID. The provided fake calibration is for tests only.

Power-upper-bound times exposure is an optical-energy upper-bound proxy, not a measurement of absorbed photons or phototoxicity. The baseline exposure also consumes the adapter's total budget. Do not claim measured photon savings without actual calibration and uncertainty.

## Safety behavior in code

The adapter is disarmed by default, rejects concurrent requests and unknown commands, checks device identity and action bounds, closes the shutter during configuration, checks exposure/focus readback, and charges a planned exposure before calling snap. On return it attempts shutter closure and setting restoration. Faults latch the adapter against continued acquisition, and restoration errors are reported rather than ignored.

Python and MMCore timeouts cannot guarantee interruption of a blocked vendor driver. Independent physical interlocks, a supervised operating protocol, and an accessible emergency stop are essential. Safe-state restoration is an attempted software action, not proof that a malfunctioning shutter actually closed.

## Integration sequence

First run the fake-core tests. Next have the instrument operator inspect `PycroAdapter` and implement any device-specific changes on a dedicated branch. Test only fixed fluorescent standards initially. Record commanded versus measured exposure, focus, gain, image scale, latency, image registration, and repeatability. Characterize repeated no-change acquisitions and any order effects.

Only then inject the calibrated adapter through `LiveProvider` into `run_episode`. The predictor, quantizer, action catalog, input preprocessing, dose units, and calibration evidence must correspond to that real acquisition system. A model trained on the packaged Gaussian-cell simulator is not a valid real-microscope calibration bundle.

Before controlled biological validation, establish independent reference labels, instrument/session-held-out evaluation, predefined thresholds and exclusion criteria, and controls for motion, axial biology, bleaching and dynamics. Log raw images outside Git, with acquisition metadata and model/source hashes.
