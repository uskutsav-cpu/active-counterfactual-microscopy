# LSFM Multi-Action Engineering Protocol

## Purpose

This dataset is used to develop and validate the generic same-specimen
multi-action optical replay infrastructure.

It is NOT used as confirmatory evidence for the biological counterfactual
verification claim.

## Dataset

Li et al. real light-sheet fluorescence microscopy defocus stacks.

Local audited test archive:

- specimens: 42
- planes per specimen: 51
- total TIFF images: 2,142
- incomplete stacks: 0
- axial spacing: 2 um
- reference in-focus plane: 26

Therefore the physical offset mapping is:

`offset_um = (plane - 26) * 2`

The complete observed range is -50 um to +50 um.

## Engineering action catalog

The first standardized catalog follows the 6-um-spaced levels used by the
authors' public autofocus implementation:

- plane 08: -36 um
- plane 11: -30 um
- plane 14: -24 um
- plane 17: -18 um
- plane 20: -12 um
- plane 23:  -6 um
- plane 26:   0 um
- plane 29:  +6 um
- plane 32: +12 um
- plane 35: +18 um
- plane 38: +24 um
- plane 41: +30 um
- plane 44: +36 um

This is a 13-action optical catalog.

## Required invariants

The stack indexer must:

1. reject unparseable TIFF names;
2. verify folder plane equals filename suffix plane;
3. reject duplicate specimen/plane pairs;
4. verify every specimen contains every expected plane;
5. hash every indexed image;
6. preserve signed physical defocus;
7. mark the reference-focus plane;
8. explicitly record that no biological task label exists.

## Scientific boundary

The LSFM study may validate:

- same-specimen indexing;
- arbitrary multi-action replay;
- signed optical intervention geometry;
- image registration/evidence extraction;
- action-selection mechanics;
- computational scaling with many candidate interventions.

The LSFM study must NOT be presented as showing that:

- a biological prediction is correct;
- H_B has been biologically validated;
- EIG is superior for biological verification;
- physical dose/time costs have been calibrated;
- prospective microscope control has been validated.

Those claims require separate data and experiments.

## Next gate

After this infrastructure passes, implement lazy multi-action replay and
action/evidence diagnostics on LSFM.

The subsequent confirmatory biological shortcut study must use untouched
data rather than tuning on BBBC006 or LSFM.
