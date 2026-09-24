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

> **Status: release candidate `0.1.0-rc1`.** This repository is the rewrite
> of the container published with the paper (`soffiafdz/hvr_cnn` 0.0.x on
> Docker Hub). On stereotaxic input it reproduces the published container's
> segmentations exactly. The image `ghcr.io/soffiafdz/hvr-cnn:0.1.0-rc1` is
> tested with Podman on Linux and on macOS (Apple silicon, emulated), not
> yet with Apptainer on a cluster. Until 0.1.0 is released, use the
> published 0.0.2 image for results you intend to publish.

## What it does

| | |
|---|---|
| Input | one T1w scan or a table of scans; MINC (`.mnc`) or NIfTI (`.nii`, `.nii.gz`) |
| Accepted spaces | already stereotaxic (ICBM152 2009c, intensity-normalised), raw / native, or AssemblyNet `mni_t1` output |
| Models | `simple`: left/right hippocampus and temporal horn. `detailed`: head, body and tail of both, plus amygdala |
| Output | label volumes in the format of the input, `volumes.tsv` (volumes and HVR per hemisphere), a QC picture per scan (on by default), `run.json` (provenance, per-scan status) |
| Hardware | CPU. Per scan, measured with the default model and QC: about 2 minutes for an already-stereotaxic scan and 6 for a raw scan (preprocessing included) on a 12-core Linux machine; about 3 and 13 on an Apple-silicon Mac with 6 CPUs given to the Podman VM |
| Runtimes | Docker, Podman, Apptainer / Singularity (HPC) |

## Quick start

Check an installation (needs no data):

```sh
podman run --rm ghcr.io/soffiafdz/hvr-cnn:0.1.0-rc1 selftest
```

The easiest way to process scans is the wrapper `bin/hvr-cnn-container`,
one shell script that mounts the files named on the command line and adds
the right options for podman, docker or apptainer:

```sh
curl -LO https://raw.githubusercontent.com/soffiafdz/hvr-cnn/main/bin/hvr-cnn-container
chmod +x hvr-cnn-container
export HVR_CNN_IMAGE=ghcr.io/soffiafdz/hvr-cnn:0.1.0-rc1
./hvr-cnn-container run -i /data/study/sub-01_T1w.nii.gz -o /data/study/hvr
./hvr-cnn-container --print run -i /data/study/sub-01_T1w.nii.gz -o /data/study/hvr   # show the command instead
```

The same run without the wrapper, with rootless Podman:

```sh
podman run --rm --read-only --network none \
    --userns=keep-id --user "$(id -u):$(id -g)" \
    --volume /data/study:/data/study \
    ghcr.io/soffiafdz/hvr-cnn:0.1.0-rc1 \
    run -i /data/study/sub-01_T1w.nii.gz -o /data/study/hvr
```

`--user` is required: without it the process runs as the image's own user
and cannot write to your folder. [Containers, explained](docs/CONTAINERS.md)
goes through every option, including Docker and Apptainer.

## Documentation

- [User guide](docs/USER_GUIDE.md): installation, inputs, outputs, every
  option, HPC usage, quality control, troubleshooting, and how to move from
  the 0.0.x container.
- [Containers, explained](docs/CONTAINERS.md): every podman / apptainer
  option used, and the pitfalls met while setting it up.
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

Code: GPL-3.0 (see `LICENSE`). Trained weights: CC BY 4.0 (cite the paper
above). Network architecture and inference code in `src/model/` by
Vladimir S. Fonov, included with permission. Third-party components: see
`NOTICE`.

No imaging data is distributed with this repository or with the image.
