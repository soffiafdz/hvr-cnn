# hvr-cnn

Automatic segmentation of the **hippocampus** and the **temporal horn of the
lateral ventricle** on T1-weighted MRI with a convolutional neural network,
and computation of the **hippocampal-to-ventricle ratio (HVR)**:

    HVR = HC / (HC + VC)

where HC is the hippocampal volume and VC the volume of the surrounding
temporal-horn CSF, per hemisphere. HVR is a marker of medial temporal
atrophy that relates more strongly to age, memory and global cognition than
hippocampal volume alone (see [Citing](#citing)).

hvr-cnn is distributed as a container image. It needs no MINC toolkit, no
Python environment and no network access on the machine that runs it.

> **Status: pre-release.** This repository is the rewrite of the container
> published with the paper (`soffiafdz/hvr_cnn` 0.0.x on Docker Hub). All
> input modes, both models, volumes/HVR, QC pictures and the output checker
> work and reproduce the published container's segmentations exactly on
> stereotaxic input; the image itself has not been built and tested yet.
> Until 0.1.0 is tagged, use the published 0.0.2 image for real work.

## What it does

| | |
|---|---|
| Input | one T1w scan or a table of scans; MINC (`.mnc`) or NIfTI (`.nii`, `.nii.gz`) |
| Accepted spaces | already stereotaxic (ICBM152 2009c, intensity-normalised), raw / native, or AssemblyNet `mni_t1` output |
| Models | `simple`: left/right hippocampus and temporal horn. `detailed`: head, body and tail of both, plus amygdala |
| Output | label volumes in the format of the input, `volumes.tsv` (volumes and HVR per hemisphere), a QC picture per scan (on by default), `run.json` (provenance, per-scan status) |
| Hardware | CPU; about one minute and 1 GB of memory per already-stereotaxic scan on 8 cores, about three minutes for a raw scan (preprocessing included) |
| Runtimes | Docker, Podman, Apptainer / Singularity (HPC) |

## Quick start

Docker or Podman:

```sh
docker run --rm -v "$PWD":/data ghcr.io/soffiafdz/hvr-cnn:latest \
    run -i /data/sub-01_T1w.nii.gz -o /data/hvr
```

Apptainer / Singularity on a cluster:

```sh
apptainer pull hvr-cnn.sif docker://ghcr.io/soffiafdz/hvr-cnn:latest
apptainer run hvr-cnn.sif run --csv scans.csv -o results
```

Check an installation (needs no data):

```sh
docker run --rm ghcr.io/soffiafdz/hvr-cnn:latest selftest
```

The same image is published as `docker.io/soffiafdz/hvr_cnn`, the location
given in the paper.

## Documentation

- [User guide](docs/USER_GUIDE.md): installation, inputs, outputs, every
  option, HPC usage, quality control, troubleshooting, and how to move from
  the 0.0.x container.
- `hvr-cnn --help` and `hvr-cnn run --help`.

## Citing

If you use hvr-cnn, please cite:

> Fernandez-Lozano S, Fonov V, Schoemaker D, Pruessner J, Potvin O,
> Duchesne S, Collins DL, for the Alzheimer's Disease Neuroimaging
> Initiative. Enhanced Detection of Age-Related and Cognitive Declines Using
> Automated Hippocampal-To-Ventricle Ratio in Alzheimer's Patients.
> *Human Brain Mapping* 2025; 46: e70265.
> https://doi.org/10.1002/hbm.70265

The analysis code of the paper lives in
[soffiafdz/hvr_validation](https://github.com/soffiafdz/hvr_validation).

## Licence

Code: GPL-3.0 (see `LICENSE`). Network architecture and inference code in
`src/model/` by Vladimir S. Fonov, included with permission. Terms for the
trained weights and third-party components: see `NOTICE`.

No imaging data is distributed with this repository or with the image.
