"""Label maps, structure volumes and HVR for both models.

The label map is chosen by model. The old container counted the `simple`
labels whatever the model was, so `detailed` always produced zero volumes
and crashed; here `detailed` HC and VC are the sums of their sub-labels.
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


def expected_labels(model: str) -> frozenset[int]:
    """All non-zero label values a segmentation of this model may contain."""
    return frozenset(
        v for side in _model(model).values() for vals in side.values() for v in vals
    )


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
    """One flat row of results for one segmentation.

    Keys: `<side>_<structure>_mm3`, `<side>_HVR`, for `detailed` also
    `<side>_<structure>_<part>_mm3`; the raw counts `*_vox` are added only
    when the voxel volume is not 1 mm^3 (otherwise they would duplicate
    `_mm3`). Raises if the
    segmentation holds a label that does not belong to the model; a missing
    label counts as zero and shows up in `missing_labels`.
    """
    labels = _model(model)
    unexpected = sorted(set(counts) - expected_labels(model))
    if unexpected:
        raise ValueError(f"labels {unexpected} do not belong to model '{model}'")

    row: dict[str, float] = {}
    unit_voxels = abs(voxel_volume_mm3 - 1.0) < 1e-9

    def put(key, vox):
        row[key + "_mm3"] = vox if unit_voxels else vox * voxel_volume_mm3
        if not unit_voxels:
            row[key + "_vox"] = vox

    for side in SIDES:
        for structure, values in labels[side].items():
            put(f"{side}_{structure}", sum(counts.get(v, 0) for v in values))
            if len(values) == len(PARTS):
                for part, v in zip(PARTS, values):
                    put(f"{side}_{structure}_{part}", counts.get(v, 0))
        row[f"{side}_HVR"] = hvr(row[f"{side}_HC_mm3"], row[f"{side}_VC_mm3"])
    row["missing_labels"] = len(expected_labels(model) - set(counts))
    return row


def _model(model: str) -> dict[str, dict[str, tuple[int, ...]]]:
    try:
        return LABELS[model]
    except KeyError:
        raise ValueError(f"unknown model '{model}', expected one of {sorted(LABELS)}") from None
