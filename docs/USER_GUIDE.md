# hvr-cnn user guide

This guide replaces the "Documentation for HVR_CNN" that accompanied the
0.0.x container published with
[Fernandez-Lozano et al., Human Brain Mapping 2025](https://doi.org/10.1002/hbm.70265).
If you arrive from that paper with an old command line, start at
[Moving from 0.0.x](#11-moving-from-00x).

> **Pre-release.** Sections marked *(planned)* describe behaviour that is
> specified but not yet in the image. Everything else can be tried today.

Contents

1. [What the tool does](#1-what-the-tool-does)
2. [Getting the image](#2-getting-the-image)
3. [First run](#3-first-run)
4. [Input](#4-input)
5. [Input spaces: which one is my scan?](#5-input-spaces-which-one-is-my-scan)
6. [Output](#6-output)
7. [Command reference](#7-command-reference)
8. [Running on an HPC cluster](#8-running-on-an-hpc-cluster)
9. [Quality control](#9-quality-control)
10. [Troubleshooting](#10-troubleshooting)
11. [Moving from 0.0.x](#11-moving-from-00x)
12. [Limitations](#12-limitations)
13. [Citing, licence, contact](#13-citing-licence-contact)

---

## 1. What the tool does

For each T1-weighted scan, hvr-cnn

1. brings the scan into the space and intensity range the network was
   trained on, if it is not there already (section 5);
2. segments each hemisphere with an ensemble of convolutional networks;
3. writes the labels, in the format of the input (MINC in, MINC out; NIfTI
   in, NIfTI out);
4. counts volumes and computes the hippocampal-to-ventricle ratio per
   hemisphere, `HVR = HC / (HC + VC)`, where HC is the hippocampus and VC
   the CSF of the temporal horn of the lateral ventricle around it;
5. renders a QC picture (skip with `--no-qc`).

HVR lies between 0 and 1. A healthy young medial temporal lobe has a large
hippocampus and a slit-like temporal horn (HVR close to 1); atrophy shrinks
the first and enlarges the second, so HVR falls.

Two models are bundled:

| `--model` | structures |
|---|---|
| `simple` (default) | hippocampus, temporal horn |
| `detailed` | hippocampus head / body / tail, temporal horn head / body / tail, amygdala |

Both give HC, VC and HVR. For `detailed`, HC and VC are the sums of their
three parts; the amygdala is reported but is not part of HVR.

## 2. Getting the image

The image is self-contained: MINC tools, PyTorch, network weights and
reference grids are inside. It never downloads anything at run time and
works without network access.

| registry | reference |
|---|---|
| GitHub Container Registry | `ghcr.io/soffiafdz/hvr-cnn:<version>` |
| Docker Hub (location printed in the paper) | `docker.io/soffiafdz/hvr_cnn:<version>` |

Pin a version (`:0.1.0`) in anything you intend to publish; `:latest`
moves. Image for `linux/amd64`; on Apple Silicon it runs under emulation.

```sh
docker pull ghcr.io/soffiafdz/hvr-cnn:latest          # Docker
podman pull ghcr.io/soffiafdz/hvr-cnn:latest          # Podman
apptainer pull hvr-cnn.sif docker://ghcr.io/soffiafdz/hvr-cnn:latest   # Apptainer / Singularity
```

Verify the installation. This needs no data and takes a few seconds:

```sh
docker run --rm ghcr.io/soffiafdz/hvr-cnn:latest selftest
apptainer run hvr-cnn.sif selftest
```

`selftest` checks the Python modules, every external program the tool
calls, the bundled weights (one forward pass per model) and that temporary
space is writable. It ends with `selftest: passed` and exit status 0.

## 3. First run

A container only sees directories you give it. With Docker and Podman that
is the `-v host_dir:container_dir` option; paths on the command line are
then paths **inside** the container.

```sh
cd /path/to/study                      # contains sub-01_T1w.nii.gz
docker run --rm -v "$PWD":/data ghcr.io/soffiafdz/hvr-cnn:latest \
    run -i /data/sub-01_T1w.nii.gz -o /data/hvr
```

Docker writes files as root unless told otherwise; add
`--user "$(id -u):$(id -g)"` so the results belong to you. Podman
(rootless) and Apptainer already run as you.

Apptainer mounts your home and current directory by default, so host paths
work unchanged:

```sh
apptainer run hvr-cnn.sif run -i sub-01_T1w.nii.gz -o hvr
```

For data elsewhere (scratch, project space) add `-B /scratch/$USER`.

Try `--dry-run` first: it validates every input, prints what would be done
and touches nothing.

## 4. Input

### 4.1 Files on the command line

```sh
hvr-cnn run -i scan1.mnc scan2.nii.gz ... -o OUTDIR
```

Accepted: `.mnc`, `.mnc.gz`, `.nii`, `.nii.gz`; 3D T1-weighted images.
DICOM is not an input format: convert first (`dcm2niix`, `dcm2mnc`).

Each scan gets an identifier, used for its output directory and file
names: the file name without extension (`sub-01_T1w.nii.gz` ->
`sub-01_T1w`). Two inputs with the same file name in different directories
would collide; hvr-cnn refuses to start and asks for a table instead.

### 4.2 A table of scans

```sh
hvr-cnn run --csv scans.csv -o OUTDIR
```

Comma-, semicolon- or tab-separated, **with a header row**:

```csv
input,subject,session,group
sub-01/ses-bl/anat/sub-01_ses-bl_T1w.nii.gz,sub-01,ses-bl,CN
sub-01/ses-m12/anat/sub-01_ses-m12_T1w.nii.gz,sub-01,ses-m12,CN
/abs/path/stx2_s02_t1.mnc,s02,,AD
```

| column | | meaning |
|---|---|---|
| `input` | required | path of the scan; relative paths are relative to the table |
| `subject` | optional | identifier becomes `<subject>` or `<subject>_<session>` |
| `session` | optional | |
| `group` | optional | copied to `volumes.tsv`, otherwise unused |

Other columns are ignored. Characters other than letters, digits, `.`, `_`
and `-` in identifiers are replaced by `-`.

### 4.3 Validation before anything runs

All inputs are checked first: existence, readability, file type, unique
identifiers. Every problem is listed at once and nothing is processed
(exit status 2):

```
2 problem(s) with the inputs:
  /data/sub-07_T1w.nii.gz: no such file
  identifier 't1' occurs 2 times; outputs would overwrite each other (use --csv with subject/session columns)
```

## 5. Input spaces: which one is my scan?

The network expects a full-head (not skull-stripped) T1w image, linearly
registered to the **ICBM152 2009c nonlinear symmetric** template on its
1 mm grid, with intensities linearly normalised to that template. A scan
in another space or intensity range gives a plausible-looking but wrong
segmentation, so hvr-cnn checks rather than assumes.

| `--input-space` | use it for | what hvr-cnn does |
|---|---|---|
| `stx` | output of a stereotaxic pipeline that matches the description above (e.g. `stx2_*_t1.mnc` of the MNI/BIC longitudinal pipeline) | verifies the geometry (1 mm, no rotation, field of view covers the medial temporal lobes), warns when the intensity range inside the medial temporal region is far from the training scale, segments |
| `native` *(planned)* | a raw scan from the scanner / `dcm2niix` / BIDS | denoising (optional), bias-field correction, linear registration to the template, intensity normalisation, resampling; then segments. Labels are written in stereotaxic **and** native space |
| `assemblynet` *(planned)* | the `mni_t1_*.nii.gz` of an AssemblyNet run | re-normalises intensities to the template (AssemblyNet's scale differs), segments; no registration needed |
| `auto` (default) | anything | decides from the geometry of each scan (`stx` when the voxel grid is the ICBM152 1 mm grid or a window of it, else `native`) and **logs the decision**; pass the explicit value if it guesses wrong |

Skull-stripped images are not supported: the network was trained with the
head present.

## 6. Output

Implemented for stereotaxic MINC input; NIfTI, native-space labels, the
QC picture and the transform are *(planned)*. File names carry the model
(`model-simple` / `model-detailed`), so both models can be run into the
same OUTDIR.

```
OUTDIR/
  volumes.tsv                      one row per scan
  run.json                         provenance and per-scan status
  <id>/
    <id>_space-stx_model-simple_seg.mnc|.nii.gz     labels on the stereotaxic grid
    <id>_space-native_model-simple_seg.mnc|.nii.gz  labels on the grid of the input (native input only)
    <id>_qc.jpg                          unless --no-qc
    <id>_to-stx.xfm                      native -> stereotaxic transform (native input only)
```

NIfTI outputs carry exactly the affine and shape of the grid they belong
to, so they overlay the input in any viewer.

### 6.1 Label values

`simple`

| value | structure |
|---|---|
| 11 / 21 | left / right hippocampus |
| 12 / 22 | left / right temporal-horn CSF |

`detailed`

| value (left / right) | structure |
|---|---|
| 111, 112, 113 / 211, 212, 213 | hippocampus tail, body, head |
| 121, 122, 123 / 221, 222, 223 | temporal-horn CSF tail, body, head |
| 130 / 230 | amygdala |

Left and right are anatomical (the subject's left).

### 6.2 `volumes.tsv`

Tab-separated, one row per scan. Identification columns (`id`, `subject`,
`session`, `group`, `model`, `input_space`, `status`) are followed by, for
each side `L` and `R`:

| column | meaning |
|---|---|
| `<side>_HC_vox`, `<side>_VC_vox` | voxels on the 1 mm stereotaxic grid |
| `<side>_HC_mm3`, `<side>_VC_mm3` | the same in mm^3 of stereotaxic space |
| `<side>_HVR` | `HC / (HC + VC)`; empty when both are zero |
| `<side>_AMY_vox`, `<side>_AMY_mm3` | amygdala (`detailed` only) |
| `<side>_HC_head_vox`, `..._body_vox`, `..._tail_vox`, same for `VC` | parts (`detailed` only) |
| `missing_labels` | expected labels that are absent. Any value but 0 makes the scan `failed` (the label file is kept for inspection) |

**Stereotaxic volumes are head-size normalised.** Linear registration to
the template scales every head to the template's size, so volumes measured
in stereotaxic space are already corrected for head size by that scaling.
This is what the paper used. HVR is a ratio of two such volumes and does
not depend on the scaling at all. Native-space volumes in mm^3 (stereotaxic
volume divided by the scaling of the transform) are *(planned)* for
`native` input.

A segmentation containing a label that does not belong to the chosen model
is an error for that scan, never a row of zeros.

### 6.3 Which model goes with which reference values

The two models are trained separately and their volumes differ
systematically (`detailed` gives a slightly larger hippocampus and a
smaller temporal horn, hence a slightly higher HVR). Do not mix them, and
use the model that the reference values you compare against were made with:

| reference | model | volumes |
|---|---|---|
| Fernandez-Lozano et al., HBM 2025 (ADNI) | `simple` (default) | stereotaxic (`*_vox` / `*_mm3`) |
| UK Biobank normative models (in preparation) | `detailed`, HC and VC as the sum of head, body and tail | native space: stereotaxic volume divided by the scale factor of the registration (`*_mm3_native`, *planned*); HVR is the same in both spaces |

### 6.4 `run.json`

Version of hvr-cnn and of PyTorch, image digest when known, device,
threads, the full command line, and for every scan: status (`ok`,
`skipped`, `failed`), the decided input space, timing, warnings and, on
failure, the error. Keep it with your results: it is the provenance record.

## 7. Command reference

```
hvr-cnn run       segment scans; write labels, volumes, HVR and QC
hvr-cnn selftest  check that this installation can run (needs no data)
hvr-cnn check     verify the outputs of a run            (planned)
hvr-cnn --version
hvr-cnn <command> --help
```

### `hvr-cnn run`

| option | default | |
|---|---|---|
| `-i, --input T1 [T1 ...]` | | scans (one of `-i` / `--csv` is required) |
| `--csv FILE` | | table of scans (section 4.2) |
| `--input-space {auto,stx,native,assemblynet}` | `auto` | section 5 |
| `-o, --output OUTDIR` | required | created if missing. There is no default location: results are never written inside the container |
| `--out-format {auto,mnc,nii}` | `auto` | `auto` = same as each input |
| `--no-qc` | QC on | skip the QC picture |
| `--overwrite` | off | redo scans whose outputs exist. Default is to skip them, so an interrupted run resumes with the same command |
| `--model {simple,detailed}` | `simple` | section 1 |
| `--denoise` | off | non-local-means denoising (native input only) |
| `--device {cpu,cuda,auto}` | `cpu` | `cuda` needs the CUDA image *(planned)* |
| `--threads N` | what the scheduler allows | honours CPU affinity / cgroups, `SLURM_CPUS_PER_TASK` and `OMP_NUM_THREADS`, never the size of the node |
| `--work-dir DIR` | temporary | intermediate files; removed afterwards unless `--keep-work` |
| `--keep-work` | off | |
| `--fail-fast` | off | stop at the first failed scan. Default: continue with the others |
| `--dry-run` | off | validate and print the plan, process nothing |
| `-v, --verbose` / `-q, --quiet` | | log every external command / warnings and errors only |

Logs go to standard error with timestamps; there are no progress bars, so
batch logs stay readable.

### Exit status

| | |
|---|---|
| 0 | success |
| 1 | the run finished but at least one scan failed; see `run.json` |
| 2 | bad command line or inputs; nothing was processed |
| 3 | `selftest` failed, or the feature is not available in this build |

### Where hvr-cnn writes

Only to `OUTDIR`, `--work-dir` and the system temporary directory
(`$TMPDIR`, else `/tmp`). The image can be mounted read-only, run as any
user and without network.

## 8. Running on an HPC cluster

Build the SIF once on a login node (compute nodes often have no internet):

```sh
module load apptainer
apptainer pull hvr-cnn_0.1.0.sif docker://ghcr.io/soffiafdz/hvr-cnn:0.1.0
apptainer run hvr-cnn_0.1.0.sif selftest
```

One job for a modest study:

```sh
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
module load apptainer
apptainer run -B "$SCRATCH" hvr-cnn_0.1.0.sif \
    run --csv "$SCRATCH/study/scans.csv" -o "$SCRATCH/study/hvr" \
        --work-dir "$SLURM_TMPDIR"
```

hvr-cnn uses the 8 CPUs SLURM granted, not the whole node. If the job hits
its time limit, resubmit it unchanged: finished scans are skipped.

For large studies split the table and use a job array, one `OUTDIR` per
task (tasks must not share an `OUTDIR`), then concatenate the
`volumes.tsv` files:

```sh
#SBATCH --array=0-19
split -d -n l/20 --additional-suffix=.csv <(tail -n +2 scans.csv) part_   # once, beforehand
# each part needs the header line again
apptainer run -B "$SCRATCH" hvr-cnn_0.1.0.sif \
    run --csv part_$(printf %02d "$SLURM_ARRAY_TASK_ID").csv -o hvr/part_$SLURM_ARRAY_TASK_ID
```

Budget roughly one to two minutes per already-stereotaxic scan on 8 cores
and 1 GB of memory; raw scans take longer because they are preprocessed.

## 9. Quality control

Always look at the QC pictures. Each scan gets one image with
axial, sagittal (left and right) and coronal rows through the medial
temporal lobe, labels overlaid on the T1.

Reject a scan when the labels are displaced from the hippocampus (failed
registration or wrong `--input-space`), when the hippocampus is clearly
truncated or leaks into the amygdala or the parahippocampal gyrus, or when
the temporal-horn label covers tissue or misses obvious CSF. Numeric hints
in `volumes.tsv`: `missing_labels` not 0, an HVR of exactly 0 or 1, or a
left/right difference far outside that of the rest of your sample.

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `no such file` for a file that exists | the container does not see that directory. Docker/Podman: mount it with `-v` and use the path inside the container. Apptainer: add `-B /that/filesystem` |
| output owned by root | Docker: add `--user "$(id -u):$(id -g)"` |
| `this looks like a command line for hvr_cnn 0.0.x` | see section 11 |
| `needs a header row with a column named 'input'` | the table has no header, or the path column has another name. 0.0.x accepted header-less tables; this version does not |
| `unsupported file type` | convert DICOM / Analyze / MGZ to NIfTI or MINC first |
| labels shifted or obviously wrong, exit status 0 | the scan was treated as `stx` but is not in ICBM152 2009c space or not intensity-normalised. Use `--input-space native` |
| very slow on a shared node | pass `--threads` equal to the CPUs you were granted |
| Apple Silicon: slow | the image is `linux/amd64` and is emulated |
| anything else | run `selftest`, then re-run the failing scan with `-v --keep-work --work-dir DIR` and open an issue with the log and `run.json` (no images, please) |

## 11. Moving from 0.0.x

Version 0.0.x (`soffiafdz/hvr_cnn:0.0.2` and earlier) stays on Docker Hub
unchanged; pin that tag to keep an old analysis reproducible. From 0.1.0
the interface is different. An old-style command line is recognised and
answered with the equivalent new options instead of a cryptic error.

| 0.0.x | now |
|---|---|
| `docker run ... soffiafdz/hvr_cnn -f in.csv -o /app/output` | `docker run ... ghcr.io/soffiafdz/hvr-cnn run --csv in.csv -o /data/out` |
| mounts had to be `/app/data` and `/app/output` | mount anything anywhere; `-o` is required |
| `--qc` opt-in | QC pictures always written; `--no-qc` to skip |
| `-f, --csv_file`; header optional, columns guessed by position | `--csv`; header required: `input[,subject,session,group]` |
| `-i, --input_img_path` (one path) | `-i` takes any number of paths |
| `-s`, `-v`, `-g` for subject, session, group | columns of the table (`-v` now means verbose) |
| `--vols`, `--hc_vols`, `--vc_vols`, `--hvr_csv` and their `--*_path` | always on: one `OUTDIR/volumes.tsv` |
| three CSV files, appended to on every run | one TSV per `OUTDIR`, written once |
| `--model detailed` crashed when volumes were requested | fixed: HC and VC are summed over head, body and tail |
| MINC only | MINC or NIfTI, output in the format of the input |
| input had to be preprocessed (`stx2`) | raw scans and AssemblyNet output accepted *(planned)* |
| a missing input was a warning, exit status 0 | all inputs validated up front, exit status 2 |
| `--clobber` | `--overwrite`; default is to skip finished scans |
| ran as root, wrote inside the image by default | any user, read-only image, no network |
| 8.6 GB image | about a third of that |

Numerical results of the `simple` model on already-stereotaxic input are
unchanged: same weights, same sampling, same label values.

## 12. Limitations

- The network was trained on T1w scans of adults, mostly older adults,
  preprocessed as described in section 5. It has not been validated on
  children, on other contrasts, or on post-surgical anatomy.
- Native input is processed cross-sectionally. The paper used a
  longitudinal pipeline with a subject-specific template; visits of one
  person processed here are registered independently.
- Volumes are stereotaxic-space volumes (section 6.2). Native-space volumes need the registration transform (section 6.3).
- CPU only for now.

## 13. Citing, licence, contact

Please cite Fernandez-Lozano et al., *Human Brain Mapping* 2025; 46:
e70265, https://doi.org/10.1002/hbm.70265.

GPL-3.0; see `LICENSE` and `NOTICE` for third-party components and the
terms of the weights. Questions and bug reports:
https://github.com/soffiafdz/hvr-cnn/issues. Do not attach imaging data
to an issue.
