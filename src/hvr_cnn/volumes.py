"""Label maps, structure volumes and HVR for both models.

The label map is chosen by model. The old container counted the `simple`
labels whatever the model was, so `detailed` always produced zero volumes
and crashed; here `detailed` HC and VC are the sums of their sub-labels.

Every volume column names its space: `_mm3_stx` on the stereotaxic grid the
network segments, `_mm3_native` in the subject's own head (the stereotaxic
volume divided by the registration's volume scaling, `scale_factor`). HVR is
the same in both spaces, since the one scaling cancels in the ratio.
"""

import math

import numpy as np

SIDES = ("L", "R")

# model -> side -> structure -> label values summed into that structure
LABELS: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {
    "simple": {
        "L": {"HC": (11,), "VC": (12,)},
        "R": {"HC": (21,), "VC": (22,)},
    },
    "detailed": {
        "L": {"HC": (111, 112, 113), "VC": (121, 122, 123), "AMY": (130,)},
        "R": {"HC": (211, 212, 213), "VC": (221, 222, 223), "AMY": (230,)},
    },
}

# Sub-label names of the `detailed` model, in label order x11, x12, x13,
# reported as extra columns. Verified on a segmentation: x13 is the anterior
# part next to the amygdala (head), x11 the posterior, superior part (tail).
# The previous implementation had these two swapped.
PARTS = ("tail", "body", "head")

STX, NATIVE = "_mm3_stx", "_mm3_native"
VOX = "_vox_stx"
SCALE_FACTOR = "scale_factor"
NO_TRANSFORM = "not computed: no transform"


def expected_labels(model: str) -> frozenset[int]:
    """All non-zero label values a segmentation of this model may contain."""
    return frozenset(
        v for side in _model(model).values() for vals in side.values() for v in vals
    )


def empty_structures(counts: dict[int, int], model: str) -> list[str]:
    """`<side>_<structure>` names none of whose labels are present."""
    return [
        f"{side}_{structure}"
        for side in SIDES
        for structure, values in _model(model)[side].items()
        if not any(counts.get(v, 0) for v in values)
    ]


def count_labels(seg: np.ndarray) -> dict[int, int]:
    """Voxel count per non-zero label value."""
    seg = np.asarray(seg)
    rounded = np.rint(seg)
    if not np.array_equal(seg, rounded):
        raise ValueError("label volume contains non-integer values")
    values, counts = np.unique(rounded.astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts) if v != 0}


def hvr(hc: float, vc: float) -> float:
    """Hippocampal-to-ventricle ratio HC / (HC + VC); NaN when both are zero."""
    if hc < 0 or vc < 0:
        raise ValueError(f"negative volume: hc={hc}, vc={vc}")
    total = hc + vc
    return hc / total if total > 0 else math.nan


def summarise(
    counts: dict[int, int], model: str, voxel_volume_mm3: float = 1.0
) -> dict[str, float]:
    """One flat row of stereotaxic results for one segmentation.

    Keys: `<side>_<structure>_mm3_stx`, `<side>_HVR`, for `detailed` also
    `<side>_<structure>_<part>_mm3_stx`; the raw counts `*_vox_stx` are
    added only when the voxel volume is not 1 mm^3 (otherwise they would
    duplicate `_mm3_stx`). Native volumes are added by `add_native`. Raises
    if the segmentation holds a label that does not belong to the model; a
    missing label counts as zero and shows up in `missing_labels`.
    """
    unexpected = sorted(set(counts) - expected_labels(model))
    if unexpected:
        raise ValueError(f"labels {unexpected} do not belong to model '{model}'")

    row: dict[str, float] = {}
    unit_voxels = abs(voxel_volume_mm3 - 1.0) < 1e-9
    for name, values in _structures(model):
        vox = sum(counts.get(v, 0) for v in values)
        row[name + STX] = vox if unit_voxels else vox * voxel_volume_mm3
        if not unit_voxels:
            row[name + VOX] = vox
    for side in SIDES:
        row[f"{side}_HVR"] = hvr(row[f"{side}_HC{STX}"], row[f"{side}_VC{STX}"])
    row["missing_labels"] = len(expected_labels(model) - set(counts))
    return _ordered(row, model)


def add_native(row: dict, scale_factor) -> dict:
    """Add `scale_factor` and the `_mm3_native` volumes to a `summarise` row.

    `scale_factor` is the volume scaling of the native -> stereotaxic linear
    registration (the product of its three scales), so stx = native x
    scale_factor. With no scaling known (None) the columns are still added,
    empty (NaN), so every row of a run has the same columns.
    """
    known = scale_factor is not None
    if known and not (math.isfinite(scale_factor) and scale_factor > 0):
        raise ValueError(f"scale factor must be a positive number, got {scale_factor}")
    row[SCALE_FACTOR] = scale_factor if known else math.nan
    for key in [k for k in row if k.endswith(STX)]:
        row[key[: -len(STX)] + NATIVE] = row[key] / scale_factor if known else math.nan
    return row


def columns(model: str) -> list[str]:
    """Every result column of a model, in table order: stereotaxic volumes and
    HVR per side, `missing_labels`, `scale_factor`, native volumes."""
    stx = []
    for side in SIDES:
        stx += [name + STX for name, _ in _structures(model, side)]
        stx.append(f"{side}_HVR")
    native = [c[: -len(STX)] + NATIVE for c in stx if c.endswith(STX)]
    return stx + ["missing_labels", SCALE_FACTOR] + native


def current_names(row: dict) -> dict:
    """A volumes.tsv row with the column names of 0.1.0-rc1/rc2 (`_mm3`,
    `_vox`, stereotaxic) renamed to the current ones (`_mm3_stx`,
    `_vox_stx`); current names and `_mm3_native` are kept as they are."""
    out = {}
    for key, value in row.items():
        if key.endswith(("_mm3", "_vox")):
            key += "_stx"
        out[key] = value
    return out


def _structures(model: str, side=None):
    """(column stem, label values) of every structure and part, in table order:
    per side, each structure followed by its parts."""
    labels = _model(model)
    for s in SIDES if side is None else (side,):
        for structure, values in labels[s].items():
            yield f"{s}_{structure}", values
            if len(values) == len(PARTS):
                for part, v in zip(PARTS, values):
                    yield f"{s}_{structure}_{part}", (v,)


def _ordered(row: dict, model: str) -> dict:
    """The row's keys in table order (`columns`), extra keys such as `_vox_stx` last."""
    order = columns(model)
    return {k: row[k] for k in order if k in row} | {k: v for k, v in row.items() if k not in order}


def _model(model: str) -> dict[str, dict[str, tuple[int, ...]]]:
    try:
        return LABELS[model]
    except KeyError:
        raise ValueError(f"unknown model '{model}', expected one of {sorted(LABELS)}") from None
