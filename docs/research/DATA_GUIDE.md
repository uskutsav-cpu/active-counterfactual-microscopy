# Real-image data guide

Sources checked: 2026-09-10. Source URLs and citations are in SOURCES.md.

## BBBC006: what it can establish

The official dataset contains repeated focal-plane acquisitions of U2OS fields. Well/site identify paired fields; wells, not individual frames, are the default split groups. Published nuclei counts come from an automated CellProfiler analysis at z16, not independent manual cell counting. The supplied labels therefore support agreement-with-reference experiments, not proof of biological truth.

Changing focus can remove genuine structure or sample different axial content. Same field identity does not prove a nuisance-only intervention. The dataset is a replay benchmark, not a substitute for controlled hardware acquisition.

The page prose describes 32 image sets while the download list runs z00 through z33. The code records observed files and checks selected planes rather than silently reconciling those statements.

The visible count-download label says ZIP, but its actual link is a CSV. The downloader uses `BBBC006_v1_counts.csv` and validates its columns.

## 1. Plan, then explicitly authorize downloads

```bash
python -m counterfactual_microscopy.research dataset-plan bbbc006 --planes 8 16 24
```

Each selected image archive is approximately 0.8 GB compressed; extraction requires additional space. The following command downloads three selected planes and the small reference-count CSV. It is never run by the installer or smoke workflow.

```bash
python -m pip install -r requirements-research-data.txt
python -m counterfactual_microscopy.research dataset-download bbbc006 --planes 8 16 24 --out data/research/bbbc006 --max-gb 3 --accept-download --extract
```

The byte limit is a download budget, not a promise of extracted disk usage. The command also checks estimated free disk. Resume validates the HTTP range and local receipts. No upstream SHA is invented; receipts are locally computed hashes plus source URLs.

## 2. Index with audited pairing and labels

```bash
python -m counterfactual_microscopy.research dataset-index bbbc006 --root data/research/bbbc006/extracted --counts-csv data/research/bbbc006/archives/BBBC006_v1_counts.csv --channel 1 --out data/research/bbbc006-manifest.csv
```

Unrecognized files and duplicates produce a report and stop indexing instead of silently dropping cases. Imagecodecs is needed to read the official LZW-compressed TIFFs. A generated TIFF fixture tests the loader's format handling; it is not a real-dataset result.

## 3. Build four disjoint panels

A minimal replay benchmark uses a fixed starting plane and two unobserved candidates:

```bash
python -m counterfactual_microscopy.research dataset-build --manifest data/research/bbbc006-manifest.csv --out data/research/bbbc006-panels --baseline z08 --actions z16 z24 --count-threshold 50 --size 64 --seed 42
python -m counterfactual_microscopy.research validate-panel data/research/bbbc006-panels/test.npz
python -m counterfactual_microscopy.research run --config configs/research/bbbc006.yaml --out results/research/bbbc006
```

The count threshold of 50 is a configurable example, not a proven best biological task. Freeze an appropriate task definition before final evaluation. Images use a common detector scale and declared linear resizing; preserve originals. Candidate planes must differ from the stored baseline. Incomplete fields require an explicit exclusion flag and are logged.

For an intentionally correlated training baseline, download at least four distinct planes and use `--second-baseline`. Example design: baseline z08 or z24, candidate z12 or z16. The training baseline–label correlation is then applied after group splitting; other splits can randomize it. Do not use the same image file as both the baseline and a purported new measurement.

## BBBC005

BBBC005 is synthetic. Its cell count is encoded in filenames, and it provides different blur conditions. Indexing treats count/sample/channel matches as *pairing candidates*. The available source description does not by itself certify that every such key is an identical latent field across all blur conditions.

Download with `dataset-download bbbc005` only after checking size. Inspect and verify the paired-field construction before using `dataset-index ... --confirm-pairing`. The flag is an explicit scientific assertion by the operator, not an automated validation. The default unconfirmed manifest cannot be replayed. Prefer the included known-latent simulator when pairing has not been established.

## Your laboratory's data

A CSV manifest must contain `specimen_id`, `group_id`, `condition`, `path`, `task_value`, `pairing_verified`, and `label_source`. Optional `sha256` records the image bytes. All views of a specimen must agree on reference metadata. Paths may be absolute or relative to the manifest.

For independent causal-hypothesis annotation, supply `reference_label` (binary) and `reference_source` and set `target: annotated_hypothesis` in the experiment config. Do not derive those annotations from the same reacquisition statistics used to verify them. The generic manifest workflow currently converts `task_value` to a binary task using a fixed threshold; multiclass, segmentation, and regression require additional predictor/metric implementations.
