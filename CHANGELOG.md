# Changelog

Changes to hvr-cnn's commands and outputs. Version 0.0.x, the container of
the Human Brain Mapping 2025 paper, is described in the user guide, section
11.

## Unreleased (0.1.0-rc3)

### Changed

- `volumes.tsv`: every volume column names its space. The stereotaxic
  volumes `<side>_<structure>[_<part>]_mm3` are now `..._mm3_stx`, and the
  voxel counts `..._vox` are `..._vox_stx`; the values are unchanged.
  Native volumes keep their name, `..._mm3_native`.
- `volumes.tsv` has the same columns for every scan of a run: the native
  volumes and `scale_factor` are always there, empty when no transform is
  known. Before, the first scan set the columns of the whole table, so a
  failed first scan dropped every volume column of the run, and a first
  scan without native volumes dropped the native volumes of all the others.
- `volumes.tsv` has Unix line endings (was `\r\n`).
- `volumes.tsv`: `input_space` is the space each scan was processed as. It
  was always `stx`.
- A scan with only some parts of a structure absent (`detailed`, e.g. a
  0-voxel hippocampal tail) stays `ok`: those parts are 0, `missing_labels`
  counts them, and the log and `run.json` carry a warning. Only a structure
  with all its labels absent fails the scan. Before, any absent label failed
  the scan and lost all its volumes.
- `run.json`: a scan's warnings are a list, `warnings` (was one string,
  `warning`, so a second warning replaced the first).
- User guide, section 6.2: stereotaxic volumes are described as a
  proportional head-size adjustment, not a neutral correction; for sex
  comparisons, native volumes residualised on ICV, or HVR (Fernandez-Lozano
  & Collins, bioRxiv 2026, doi:10.64898/2026.04.13.718285). Section 6.3
  labels the reference values by space: stereotaxic for the HBM 2025 ADNI
  results, native for the UK Biobank norms.

### Added

- `volumes.tsv`: `scale_factor`, the volume scaling of the native ->
  stereotaxic registration (stereotaxic = native x `scale_factor`).
- `--csv`: optional `xfm` column, the native -> stereotaxic transform of a
  stereotaxic input kept elsewhere than next to the scan. Checked up front
  with the inputs; the container wrapper mounts it.
- When native volumes cannot be computed (stereotaxic or AssemblyNet input
  without a transform): a warning per scan, `"native_volumes": "not
  computed: no transform"` in `run.json`, and a warning at the end of the
  run that counts those scans.
- `hvr-cnn check --reference` compares against runs of 0.1.0-rc1/rc2,
  reading their columns under the new names. A stereotaxic volume missing
  from the reference is now a failure; it was skipped without a message.

### Column names, rc1/rc2 to rc3

| rc1, rc2 | rc3 |
|---|---|
| `L_HC_mm3`, `L_VC_mm3`, `L_AMY_mm3`, `L_HC_head_mm3`, ... | `L_HC_mm3_stx`, `L_VC_mm3_stx`, `L_AMY_mm3_stx`, `L_HC_head_mm3_stx`, ... |
| `L_HC_vox`, ... | `L_HC_vox_stx`, ... |
| `L_HC_mm3_native`, ... | unchanged |
| `L_HVR`, `R_HVR`, `missing_labels` | unchanged |
| | `scale_factor` (new) |

Same for `R_`. The user guide (section 11) has a one-line `awk` command
that renames the columns of an existing table.
