"""Native T1w -> stereotaxic volume on the ICBM152 2009c grid.

Mirrors the T1 preprocessing of the longitudinal pipeline the network was
trained on (`t1preprocessing_v10`, non-SynthStrip branch), with its
parameters, minus everything the network does not need:

1. optional NLM denoising (sigma = noise estimate x 0.7, patch 3, search 1)
2. winsorize 1-95 % -> Otsu head mask -> defrag -> expand 50 mm, close
3. linear registration to the template, NMI, masked, four minctracc stages
   (lsq6 8 mm, lsq7 8 mm, lsq9 4 mm, lsq9 2 mm reversed), the last two
   nine-parameter: the pipeline's `bestlinreg_20180117` configuration
4. template brain mask back-projected -> N4 (200 x 4, B-spline 200 mm, or
   50 mm when `--n4-distance 50`), weight = head mask x brain mask
5. `volume_pol --order 1` to the template within the masks
6. `itk_resample --order 4` onto the template grid

Cross-sectional: no subject-specific template, unlike the pipeline's stx2.
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np

from . import minctools
from .selftest import ASSETS

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(os.environ.get("HVR_CNN_TEMPLATE_DIR", str(ASSETS)))
TEMPLATE_T1 = "mni_icbm152_t1_tal_nlin_sym_09c.mnc"
TEMPLATE_MASK = "mni_icbm152_t1_tal_nlin_sym_09c_mask.mnc"

# `bestlinreg_20180117`: (parameters, blur fwhm, step, tolerance, simplex, reversed)
STAGES = (
    ("-lsq6", 8, 4, 0.0001, 16, False),
    ("-lsq7", 8, 4, 0.0001, 16, False),
    ("-lsq9", 4, 4, 0.0001, 8, False),
    ("-lsq9", 2, 2, 0.0000001, 4, True),
)


class PreprocessError(RuntimeError):
    pass


def template_files():
    t1, mask = TEMPLATE_DIR / TEMPLATE_T1, TEMPLATE_DIR / TEMPLATE_MASK
    for f in (t1, mask):
        if not f.is_file():
            raise PreprocessError("template file missing: %s (set HVR_CNN_TEMPLATE_DIR)" % f)
    return t1, mask


def _stat(path, *args):
    return float(minctools.run(["mincstats", "-q", path] + list(args)).stdout.split()[0])


def _com(path):
    return [float(v) for v in minctools.run(["mincstats", "-q", path, "-com", "-world_only"]).stdout.split()[:3]]


def fix_input(t1, out):
    """Float MINC in z-y-x order with positive steps and a canonical header.

    Replaces the pipeline's `convert_and_fix`. Whatever converter wrote the
    input (dcm2mnc: negative steps, per-slice ranges; NIfTI: float32
    affine), the volume the registration sees is byte-for-byte the same, so
    MINC and NIfTI inputs of the same scan give the same result.
    """
    from . import io

    data, affine = io.read_minc(t1, np.float32)
    return io.write_minc(data, affine, out, decimals=3)


def denoise(t1, out):
    sigma = float(minctools.run(["noise_estimate", t1]).stdout.split()[0])
    minctools.run(["itk_minc_nonlocal_filter", t1, out, "--clobber", "--patch", "3", "--search", "1",
                   "--sigma", str(sigma * 0.7)])
    return out


def head_mask(t1, work):
    """Winsorized T1 and a closed head mask on its grid, plus the raw Otsu mask."""
    lo, hi = _stat(t1, "-pctT", "1"), _stat(t1, "-pctT", "95")
    trunc = work / "trunc.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-expression", "clamp(A[0],%g,%g)" % (lo, hi), t1, trunc])
    otsu = work / "otsu.mnc"
    minctools.run(["itk_morph", trunc, otsu, "--bimodal"])
    defrag = work / "otsu_defrag.mnc"
    minctools.run(["mincdefrag", otsu, defrag, "1", "6"])
    expanded = work / "otsu_expanded.mnc"
    minctools.run(["autocrop", "-clobber", "-isoexpand", "50mm", defrag, expanded])
    closed_big = work / "otsu_closed_big.mnc"
    minctools.run(["itk_morph", expanded, closed_big, "--exp", "D[25] E[25]"])
    closed = work / "otsu_closed.mnc"
    minctools.run(["itk_resample", closed_big, closed, "--clobber", "--labels", "--byte", "--like", trunc])
    return trunc, closed, defrag


def register(source, target, out_xfm, work, source_mask=None, target_mask=None):
    """Four-stage minctracc, NMI; centre-of-mass initialisation."""
    prev = None
    blurred = {}

    def blur(path, fwhm, tag):
        key = (str(path), fwhm)
        if key not in blurred:
            out = work / ("%s_blur%d.mnc" % (tag, fwhm))
            minctools.run(["fast_blur", path, out, "--fwhm", str(fwhm), "--clobber"])
            blurred[key] = out
        return blurred[key]

    for i, (params, fwhm, step, tol, simplex, reverse) in enumerate(STAGES):
        src, trg = blur(source, fwhm, "src"), blur(target, fwhm, "trg")
        xfm = work / ("reg_stage%d.xfm" % i)
        if reverse:
            src, trg = trg, src
        args = ["minctracc", src, trg, "-clobber", params, "-nmi", "-simplex", str(simplex), "-tol", str(tol),
                "-step", str(step), str(step), str(step)]
        if prev is not None:
            init = prev
            if reverse:
                init = work / ("reg_stage%d_init.xfm" % i)
                minctools.run(["xfminvert", "-clobber", prev, init])
            args += ["-transformation", init]
        else:
            cs, ct = _com(source), _com(target)
            init = work / "reg_init.xfm"
            minctools.run(["param2xfm", "-clobber", "-translation"] + [str(ct[k] - cs[k]) for k in range(3)] + [init])
            args += ["-transformation", init]
        if reverse:
            if source_mask is not None:
                args += ["-model_mask", source_mask]
        else:
            if source_mask is not None:
                args += ["-source_mask", source_mask]
            if target_mask is not None:
                args += ["-model_mask", target_mask]
        args.append(xfm)
        minctools.run(args)
        if reverse:
            fwd = work / ("reg_stage%d_sol.xfm" % i)
            minctools.run(["xfminvert", "-clobber", xfm, fwd])
            xfm = fwd
        prev = xfm
    shutil.copyfile(str(prev), str(out_xfm))
    return out_xfm


def native_to_stx(t1, out_stx, out_xfm, work, do_denoise=False, n4_distance=200):
    """Produce the stereotaxic, intensity-normalised volume and the native->stx xfm."""
    work = Path(work)
    tpl_t1, tpl_mask = template_files()
    fixed = fix_input(t1, work / "input_fixed.mnc")
    if do_denoise:
        fixed = denoise(fixed, work / "input_nlm.mnc")
    trunc, closed, defrag = head_mask(fixed, work)
    masked_trunc = work / "trunc_masked.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-expression", "A[0]*A[1]", trunc, closed, masked_trunc])
    masked = work / "input_masked.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-expression", "A[0]*A[1]", fixed, closed, masked])
    register(masked_trunc, tpl_t1, out_xfm, work)
    brain = work / "brainmask_native.mnc"
    minctools.run(["itk_resample", tpl_mask, brain, "--clobber", "--labels", "--byte", "--like", defrag,
                   "--transform", out_xfm, "--invert_transform"])
    weight = work / "weightmask.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-byte", "-expression", "A[0]*A[1]", defrag, brain, weight])
    n4 = work / "n4.mnc"
    minctools.run(["N4BiasFieldCorrection", "-d", "3", "-i", masked, "--rescale-intensities", "1",
                   "--bspline-fitting", str(n4_distance), "--output", "[%s,%s]" % (n4, work / "n4_field.mnc"),
                   "--mask-image", closed, "--weight-image", weight,
                   "--convergence", "[200x200x200x200,0.0]"])
    n4_short = work / "n4_short.mnc"
    minctools.run(["mincreshape", "-q", "-clobber", "-short", n4, n4_short])
    expfile = work / "volpol.exp"
    minctools.run(["volume_pol", n4_short, tpl_t1, "--order", "1", "--expfile", expfile, "--noclamp", "--clob",
                   "--source_mask", weight, "--target_mask", tpl_mask])
    expression = open(str(expfile)).read().strip()
    clp = work / "clp.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-expression", expression, n4_short, clp, "-zero", "-short"])
    minctools.run(["itk_resample", clp, out_stx, "--clobber", "--order", "4", "--like", tpl_t1,
                   "--transform", out_xfm, "--short"])
    log.debug("native->stx done: %s", out_stx)
    return out_stx, out_xfm


def xfm_scale_factor(xfm):
    """Product of the three scales of a linear xfm (native -> stx volume ratio)."""
    out = minctools.run(["xfm2param", xfm]).stdout
    for line in out.splitlines():
        if line.startswith("-scale"):
            sx, sy, sz = (float(v) for v in line.split()[1:4])
            return sx * sy * sz
    raise PreprocessError("cannot read scales from %s" % xfm)
