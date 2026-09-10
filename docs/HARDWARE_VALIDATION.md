# Hardware validation plan

## Objective
Validate that commanded optical interventions are accurate, repeatable, sufficiently fast, and biologically interpretable before using them as counterfactual tests.

## Candidate intervention families

| Intervention | Primary nuisance targeted | Main confounds to measure |
|---|---|---|
| Illumination intensity | brightness / photon statistics | bleaching, phototoxicity, nonlinearity |
| Exposure time | SNR / saturation | motion blur, bleaching, timing changes |
| Detector gain | electronic amplification | read noise, clipping, device-specific transforms |
| Axial focus | blur / PSF mismatch | true 3D structure change, stage backlash |
| Illumination angle/pattern | shading / contrast transfer | spatially varying dose |
| Excitation wavelength/channel | spectral dependence | fluorophore specificity, crosstalk |

## Validation checklist

### Command accuracy
- Log requested and reported hardware settings.
- Measure setting repeatability over repeated cycles.
- Record command-to-exposure latency and jitter.

### Optical response
- Use stable fluorescent standards where appropriate.
- Characterize intensity response, saturation, noise, and PSF changes.
- Measure field nonuniformity and channel registration.

### Registration
- Estimate translation/rotation/deformation between baseline and reacquisition.
- Establish a preregistered registration-quality threshold.
- Abstain when registration fails rather than forcing a verdict.

### Dose
- Define a dose proxy tied to illumination power and exposure duration.
- Validate whether gain-only changes alter biological dose.
- Track cumulative dose across verification steps.

### Temporal confounding
- Measure no-intervention repeated acquisitions to quantify natural drift.
- Randomize intervention order in calibration studies.
- Bound the time window within which biology is assumed stable.

## Safety boundary in software

Every real adapter should enforce parameter bounds, action validation, timeouts, and a dry-run mode. The research policy must not be able to issue arbitrary device commands.
