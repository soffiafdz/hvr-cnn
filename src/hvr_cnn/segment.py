"""Segmentation of a stereotaxic T1w MINC volume: per hemisphere, resample
onto the model's reference grid (left side X-flipped), patch-wise inference,
resample the labels back onto the input grid with relabelling, merge L+R.

The operations and their parameters are those of the original
implementation; the model is loaded once per run and the device is a
parameter.
"""

import logging
import math
from pathlib import Path

import torch

from . import minctools
from .selftest import ASSETS, MODELS

log = logging.getLogger(__name__)

PATCH = 96
STRIDE = 32
CROP = 8
STX_TOLERANCE = 1e-3  # mm, on step sizes
ICBM_START = (-96.0, -132.0, -78.0)  # ICBM152 2009c 1 mm grid origin, x y z
# 90th percentile of the intensities inside the reference box: 77-96 on 13
# pipeline stx2 volumes, 188 on a raw scan. Outside this range -> warning.
INTENSITY_P90_RANGE = (50.0, 140.0)

# network output class -> label value, per model and side
REMAP = {
    "simple": {"L": {1: 11, 2: 12}, "R": {1: 21, 2: 22}},
    "detailed": {
        "L": {1: 111, 2: 112, 3: 113, 4: 121, 5: 122, 6: 123, 7: 130},
        "R": {1: 211, 2: 212, 3: 213, 4: 221, 5: 222, 6: 223, 7: 230},
    },
}


class GeometryError(ValueError):
    """The input is not on a stereotaxic grid the model can be applied to."""


def read_geometry(path):
    """(lengths, starts, steps) in x, y, z order, and whether the direction cosines are identity."""
    from minc2_simple import minc2_file

    f = minc2_file(str(path))
    f.setup_standard_order()
    dims = f.representation_dims()
    f.close()
    lengths = tuple(int(d.length) for d in dims)
    starts = tuple(float(d.start) for d in dims)
    steps = tuple(float(d.step) for d in dims)
    identity = all(
        abs(float(d.dir_cos[i]) - (1.0 if i == k else 0.0)) < 1e-6
        for k, d in enumerate(dims) for i in range(3)
    ) if all(getattr(d, "have_dir_cos", 0) for d in dims) else True
    return lengths, starts, steps, identity


def check_stx_geometry(path, reference=ASSETS / MODELS["simple"][1]):
    """Raise GeometryError unless `path` is 1 mm, axis-aligned and covers the reference box."""
    lengths, starts, steps, identity = read_geometry(path)
    problems = []
    if any(abs(abs(s) - 1.0) > STX_TOLERANCE for s in steps):
        problems.append("voxel size is %s mm, expected 1 mm" % (tuple(round(s, 3) for s in steps),))
    if not identity:
        problems.append("direction cosines are not identity (oblique volume)")
    ref_len, ref_start, ref_step, _ = read_geometry(reference)
    for axis, (n, s0, st, rn, rs0, rst) in enumerate(zip(lengths, starts, steps, ref_len, ref_start, ref_step)):
        lo, hi = sorted((s0, s0 + st * (n - 1)))
        rlo, rhi = sorted((rs0, rs0 + rst * (rn - 1)))
        if lo > rlo + STX_TOLERANCE or hi < rhi - STX_TOLERANCE:
            problems.append("%s covers %.0f..%.0f mm, the model needs %.0f..%.0f" % ("xyz"[axis], lo, hi, rlo, rhi))
    if problems:
        raise GeometryError(
            "%s is not a usable stereotaxic volume (ICBM152 2009c, 1 mm): %s" % (path, "; ".join(problems)))
    return lengths, starts, steps


def decide_space(path):
    """'stx' if the grid is the ICBM152 1 mm grid or an axis-aligned window of
    it (voxel centres on integer mm from the template origin), else 'native'.
    Returns (space, reason)."""
    lengths, starts, steps, identity = read_geometry(path)
    if any(abs(abs(s) - 1.0) > STX_TOLERANCE for s in steps):
        return "native", "voxel size %s mm is not 1 mm" % (tuple(round(s, 3) for s in steps),)
    if not identity:
        return "native", "oblique volume (direction cosines not identity)"
    offsets = [abs((s0 - o) - round(s0 - o)) for s0, o in zip(starts, ICBM_START)]
    if max(offsets) > 0.01:
        return "native", "grid origin %s is not aligned with the ICBM152 grid" % (tuple(round(s, 3) for s in starts),)
    try:
        check_stx_geometry(path)
    except GeometryError as exc:
        return "native", str(exc).split(": ", 1)[-1]
    return "stx", "1 mm grid aligned with ICBM152 %s, starts %s" % (
        "x".join(map(str, lengths)), tuple(int(round(s)) for s in starts))


def intensity_p90(path, reference=ASSETS / MODELS["simple"][1]):
    """90th percentile of the input inside the reference box (training scale: ~80)."""
    import numpy as np
    from minc2_simple import minc2_file

    f = minc2_file(str(path))
    f.setup_standard_order()
    data = np.asarray(f.load_complete_volume("float64"))
    dims = f.representation_dims()
    f.close()
    ref_len, ref_start, ref_step, _ = read_geometry(reference)
    index = []
    for d, n, s0, st in zip(dims, ref_len, ref_start, ref_step):
        lo, hi = sorted((s0, s0 + st * (n - 1)))
        i0 = int(round((lo - d.start) / d.step))
        i1 = int(round((hi - d.start) / d.step))
        a, b = sorted((i0, i1))
        index.append(slice(max(a, 0), min(b + 1, int(d.length))))
    box = data[index[2], index[1], index[0]]  # array order z, y, x
    return float(np.percentile(box, 90)) if box.size else float("nan")


def flip_xfm(work_dir):
    """The left-right flip used to put the left hemisphere on the (right-side) reference grid."""
    xfm = Path(work_dir) / "flip_x.xfm"
    if not xfm.exists():
        minctools.run(["param2xfm", "-clobber", "-scales", "-1", "1", "1", xfm])
    return xfm


def load_volume(path):
    from minc2_simple import minc2_file

    f = minc2_file(str(path))
    f.setup_standard_order()
    data = f.load_complete_volume_tensor(minc2_file.MINC2_FLOAT)
    f.close()
    return data


def save_labels_like(labels, like, path):
    """Write a byte label volume with the sampling and metadata of `like`."""
    from minc2_simple import minc2_file

    src = minc2_file(str(like))
    out = minc2_file()
    out.define(src.store_dims(), minc2_file.MINC2_BYTE, minc2_file.MINC2_BYTE)
    out.create(str(path))
    out.setup_standard_order()
    out.copy_metadata(src)
    out.save_complete_volume_tensor(labels.to(torch.int8).contiguous())
    out.close()
    src.close()


def infer(volume, model, device="cpu", patch=PATCH, stride=STRIDE, crop=CROP):
    """Patch-wise inference over a 3D tensor; returns a 3D int64 label tensor.

    Same accumulation as the original `segment_with_patches_overlap`:
    log-softmax averaged over overlapping (cropped) patches, then argmax.
    """
    data = volume.unsqueeze(0).unsqueeze(0)
    size = data.shape[2:]
    inner = patch - 2 * crop
    roi = [n - 2 * crop for n in size]
    fuzzy = None
    weight = torch.zeros(1, 1, *size)
    ones = torch.ones(1, 1, inner, inner, inner)
    model = model.to(device)
    with torch.no_grad():
        for k in range(math.ceil((roi[0] - crop) / stride)):
            for l in range(math.ceil((roi[1] - crop) / stride)):
                for m in range(math.ceil((roi[2] - crop) / stride)):
                    c = [k * stride + crop, l * stride + crop, m * stride + crop]
                    for i in range(3):
                        c[i] = max(min(c[i], size[i] - patch + crop - 1), crop)
                    block = data[:, :, c[0] - crop: c[0] - crop + patch,
                                 c[1] - crop: c[1] - crop + patch,
                                 c[2] - crop: c[2] - crop + patch]
                    out = torch.log_softmax(model(block.to(device))["seg"], 1).cpu()
                    if fuzzy is None:
                        fuzzy = torch.zeros(1, out.shape[1], *size)
                    sl = (slice(None), slice(None), slice(c[0], c[0] + inner),
                          slice(c[1], c[1] + inner), slice(c[2], c[2] + inner))
                    fuzzy[sl] += out[:, :, crop: crop + inner, crop: crop + inner, crop: crop + inner]
                    weight[sl] += ones
        invalid = weight < 1.0
        weight.masked_fill_(invalid, 1.0)
        fuzzy /= weight
        fuzzy = torch.softmax(fuzzy, 1)
        fuzzy[:, 0:1].masked_fill_(invalid, 1.0)
        fuzzy[:, 1:].masked_fill_(invalid, 0.0)
        labels = fuzzy.max(1)[1]
        labels.masked_fill_(invalid[:, 0], 0)
    return labels[0]


def segment_hemisphere(t1, side, model_name, model, work_dir, device="cpu", tag=""):
    """Resample `t1` onto the reference grid (flipped for the left side), run
    the network, and return the labels resampled back onto `t1`'s grid with
    the side's final label values."""
    work_dir = Path(work_dir)
    reference = ASSETS / MODELS[model_name][1]
    transform = ["--transform", flip_xfm(work_dir)] if side == "L" else []
    side_t1 = work_dir / ("%s_%s_ref.mnc" % (tag, side))
    side_seg = work_dir / ("%s_%s_seg.mnc" % (tag, side))
    side_out = work_dir / ("%s_%s_labels.mnc" % (tag, side))
    minctools.run(["itk_resample", t1, side_t1, "--clobber", "--order", "4", "--like", reference] + transform)
    labels = infer(load_volume(side_t1), model, device)
    save_labels_like(labels, side_t1, side_seg)
    lut = ";".join("%d %d" % kv for kv in REMAP[model_name][side].items())
    minctools.run(["itk_resample", side_seg, side_out, "--clobber", "--labels", "--byte",
                   "--like", t1, "--lut-string", lut] + transform)
    return side_out


def segment_stx(t1, output, model_name, model, work_dir, device="cpu"):
    """Full segmentation of a stereotaxic MINC volume into `output` (byte labels, both sides)."""
    t1, output = Path(t1), Path(output)
    check_stx_geometry(t1)
    tag = t1.name.split(".")[0]
    sides = [segment_hemisphere(t1, s, model_name, model, work_dir, device, tag) for s in ("L", "R")]
    minctools.run(["minccalc", "-copy_header", "-q", "-clobber", "-labels", "-byte",
                   "-express", "A[0]>0?A[0]:A[1]", sides[0], sides[1], output])
    return output
