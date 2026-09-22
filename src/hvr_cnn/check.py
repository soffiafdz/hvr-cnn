"""`hvr-cnn check OUTDIR [--reference DIR]`: verify the outputs of a run.

Runs inside the container, so it can read MINC and NIfTI. Checks, per scan
recorded as `ok` in `run.json`: the label file exists and holds exactly the
model's label set, HVR is within (0, 1), NIfTI outputs sit on the input's
grid (when the input is reachable), and, with `--reference`, that the
labels match those of a reference run (Dice per label) and the volumes
agree within a tolerance.
"""

import csv
import json
import logging
import math
from pathlib import Path

import numpy as np

from . import io, volumes

log = logging.getLogger(__name__)

DICE_MIN = 0.99
VOLUME_TOLERANCE = 0.01  # relative


def load_labels(path):
    path = str(path)
    if path.endswith(".mnc"):
        data, affine = io.read_minc(path)
        return data, affine
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.affine


def read_run(outdir):
    outdir = Path(outdir)
    run = json.load(open(str(outdir / "run.json")))
    rows = {}
    with open(str(outdir / "volumes.tsv"), newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["id"]] = row
    return run, rows


def dice_per_label(a, b, labels):
    out = {}
    for lab in labels:
        x, y = a == lab, b == lab
        denom = x.sum() + y.sum()
        out[lab] = 2.0 * np.logical_and(x, y).sum() / denom if denom else math.nan
    return out


def check_run(outdir, reference=None):
    """Yield (ok, message) for every check; the caller counts failures."""
    outdir = Path(outdir)
    try:
        run, rows = read_run(outdir)
    except (OSError, ValueError) as exc:
        yield False, "cannot read run.json / volumes.tsv in %s: %s" % (outdir, exc)
        return
    model = run["arguments"]["model"]
    expected = volumes.expected_labels(model)
    ref_run = ref_rows = None
    if reference:
        try:
            ref_run, ref_rows = read_run(reference)
        except (OSError, ValueError) as exc:
            yield False, "cannot read the reference run in %s: %s" % (reference, exc)
            return
        yield (ref_run["arguments"]["model"] == model,
               "reference model %s vs %s" % (ref_run["arguments"]["model"], model))

    n_ok = 0
    for scan in run["scans"]:
        sid = scan["id"]
        if scan.get("status") not in ("ok", "skipped"):
            yield True, "%s: status %s (not checked)" % (sid, scan.get("status"))
            continue
        n_ok += 1
        label_path = Path(scan["outputs"][0])
        if not label_path.is_file():
            yield False, "%s: missing %s" % (sid, label_path)
            continue
        data, affine = load_labels(label_path)
        found = set(volumes.count_labels(data))
        yield found == expected, "%s: label set %s" % (sid, "complete" if found == expected else
                                                      "wrong: found %s expected %s" % (sorted(found), sorted(expected)))
        row = rows.get(sid)
        if row is None:
            yield False, "%s: no row in volumes.tsv" % sid
            continue
        for side in volumes.SIDES:
            try:
                h = float(row["%s_HVR" % side])
            except (KeyError, ValueError):
                h = math.nan
            yield 0.0 < h < 1.0, "%s: %s_HVR = %s" % (sid, side, row.get("%s_HVR" % side, "missing"))
        if str(label_path).endswith((".nii", ".nii.gz")):
            inp = Path(scan["input"])
            if inp.is_file() and str(inp).endswith((".nii", ".nii.gz")):
                import nibabel as nib

                a, b = nib.load(str(inp)), nib.load(str(label_path))
                same = a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-3)
                yield same, "%s: NIfTI output on the input grid" % sid + ("" if same else " - NO")
            else:
                yield True, "%s: input not reachable, grid check skipped" % sid
        if ref_rows is not None:
            ref_scan = next((s for s in ref_run["scans"] if s["id"] == sid), None)
            if ref_scan is None or ref_scan.get("status") not in ("ok", "skipped"):
                yield False, "%s: not in the reference run" % sid
                continue
            ref_data, _ = load_labels(ref_scan["outputs"][0])
            if ref_data.shape != data.shape:
                yield False, "%s: reference grid %s differs from %s" % (sid, ref_data.shape, data.shape)
                continue
            dice = dice_per_label(data, ref_data, sorted(expected))
            worst = min(dice.values())
            yield worst >= DICE_MIN, "%s: Dice vs reference min %.4f (%s)" % (
                sid, worst, " ".join("%d:%.3f" % kv for kv in dice.items()))
            ref_row = ref_rows.get(sid, {})
            for key in (k for k in row if k.endswith(("_mm3", "_vox"))):
                try:
                    v, r = float(row[key]), float(ref_row[key])
                except (KeyError, ValueError):
                    continue
                rel = abs(v - r) / r if r else (0.0 if v == 0 else math.inf)
                yield rel <= VOLUME_TOLERANCE, "%s: %s %g vs reference %g (%.2f%%)" % (sid, key, v, r, 100 * rel)
    yield n_ok > 0, "%d scan(s) with results" % n_ok


def main(outdir, reference=None):
    failures = 0
    for ok, message in check_run(outdir, reference):
        log.log(logging.INFO if ok else logging.ERROR, "%-4s %s", "ok" if ok else "FAIL", message)
        failures += not ok
    log.log(logging.INFO if not failures else logging.ERROR,
            "check: %s", "passed" if not failures else "%d failure(s)" % failures)
    return failures
