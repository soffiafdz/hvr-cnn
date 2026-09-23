"""Native T1w -> stereotaxic volume on the ICBM152 2009c grid.

A minimal, cross-sectional version of the T1 preprocessing of the MNI
longitudinal pipeline (NIST-MNI nist_mni_pipelines 0.2.00, the version
that produced the UK Biobank `stx2` volumes, and the same N3 + NLM recipe
as the network's training data), with that pipeline's parameters:

1. NLM denoising (sigma = noise estimate x 0.7, patch 3, search 1); on by
   default, as in the pipeline container
2. winsorize 1-95 % -> Otsu head mask -> defrag -> expand 50 mm, close
3. linear registration to the template, NMI, masked, four minctracc stages
   (lsq6 8 mm, lsq7 8 mm, lsq9 4 mm, lsq9 2 mm reversed): the pipeline's
   `bestlinreg_20180117` configuration
4. template brain mask back-projected to native space (the pipeline uses
   a SynthStrip mask here)
5. N3 (`nu_estimate -stop 0.00001 -fwhm 0.1 -iterations 1000`, masked,
   B-spline distance 50 mm at 3 T, the tool default of 200 mm at 1.5 T) ->
   `volume_pol` gain to the template
6. second N3 pass -> `volume_pol --order 1` within brain / template masks
   (the pipeline's `clp2` step)
7. `itk_resample --order 4` onto the template grid

Cross-sectional: no subject-specific linear template, unlike `stx2`.
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


def n3(image, out, mask, field_strength=3.0):
    """N3 bias-field correction with the pipeline's settings (`minc_tools.nu_correct`)."""
    imp = Path(str(out) + ".imp")
    cmd = ["nu_estimate", "-clobber", "-stop", "0.00001", "-fwhm", "0.1", "-iterations", "1000",
           image, imp, "-mask", mask]
    if field_strength >= 3.0:
        cmd += ["-distance", "50"]
    minctools.run(cmd)
    minctools.run(["nu_evaluate", "-clobber", image, "-mapping", imp, out, "-mask", mask])
    return out


def intensity_gain(image, out, source_mask=None, target_mask=None):
    """`volume_pol --order 1` to the template, applied with minccalc (pipeline `minc.volume_pol`)."""
    tpl_t1, _ = template_files()
    expfile = Path(str(out) + ".exp")
    cmd = ["volume_pol", image, tpl_t1, "--order", "1", "--expfile", expfile, "--noclamp", "--clob"]
    if source_mask is None:  # the pipeline masks out NaNs only
        source_mask = Path(str(out) + "_valid.mnc")
        minctools.run(["minccalc", "-q", "-clobber", "-byte", "-labels", "-express", "!isnan(A[0])", image,
                       source_mask])
    cmd += ["--source_mask", source_mask]
    if target_mask is not None:
        cmd += ["--target_mask", target_mask]
    minctools.run(cmd)
    expression = open(str(expfile)).read().strip()
    minctools.run(["minccalc", "-q", "-clobber", "-expression", expression, image, out, "-zero", "-short"])
    return out, expression


def native_to_stx(t1, out_stx, out_xfm, work, do_denoise=True, field_strength=3.0):
    """Produce the stereotaxic, intensity-normalised volume and the native->stx xfm."""
    work = Path(work)
    tpl_t1, tpl_mask = template_files()
    fixed = fix_input(t1, work / "input_fixed.mnc")
    if do_denoise:
        fixed = denoise(fixed, work / "input_nlm.mnc")
    trunc, closed, defrag = head_mask(fixed, work)
    masked_trunc = work / "trunc_masked.mnc"
    minctools.run(["minccalc", "-q", "-clobber", "-expression", "A[0]*A[1]", trunc, closed, masked_trunc])
    register(masked_trunc, tpl_t1, out_xfm, work)
    brain = work / "brainmask_native.mnc"
    minctools.run(["itk_resample", tpl_mask, brain, "--clobber", "--labels", "--byte", "--like", fixed,
                   "--transform", out_xfm, "--invert_transform"])
    # first pass (pipeline `clp`): N3, then a gain to the template
    n3_1 = n3(fixed, work / "n3_1.mnc", brain, field_strength)
    clp, _ = intensity_gain(n3_1, work / "clp.mnc")
    # second pass (pipeline `clp2`): N3 again, gain within the brain masks
    n3_2 = n3(clp, work / "n3_2.mnc", brain, field_strength)
    clp2, _ = intensity_gain(n3_2, work / "clp2.mnc", source_mask=brain, target_mask=tpl_mask)
    minctools.run(["itk_resample", clp2, out_stx, "--clobber", "--order", "4", "--like", tpl_t1,
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


def assemblynet_mask_for(mni_t1):
    """The `mni_mask_*` sibling of an AssemblyNet `mni_t1_*` file, or None."""
    mni_t1 = Path(mni_t1)
    if not mni_t1.name.startswith("mni_t1_"):
        return None
    mask = mni_t1.with_name("mni_mask_" + mni_t1.name[len("mni_t1_"):])
    return mask if mask.is_file() else None


def assemblynet_to_stx(mni_t1, mni_mask, out_stx, work):
    """AssemblyNet `mni_t1` (already affine-registered to MNI, own intensity
    scale) -> the training intensity scale, on AssemblyNet's grid: linear
    `volume_pol` to the template within the brain masks. No registration."""
    work = Path(work)
    tpl_t1, tpl_mask = template_files()
    mask = work / "asm_mask.mnc"
    minctools.run(["mincreshape", "-q", "-clobber", "-byte", mni_mask, mask])
    _, expression = intensity_gain(mni_t1, out_stx, source_mask=mask, target_mask=tpl_mask)
    log.debug("assemblynet normalisation: %s", expression)
    return out_stx, expression


def stx_xfm_for(stx_t1):
    """The native->stx transform stored next to a stereotaxic T1 by the
    longitudinal pipeline (`stx2_<id>_t1.mnc` -> `stx2_<id>_t1.xfm`), or None."""
    stx_t1 = Path(stx_t1)
    name = stx_t1.name
    for ext in (".mnc.gz", ".mnc", ".nii.gz", ".nii"):
        if name.endswith(ext):
            candidate = stx_t1.with_name(name[: -len(ext)] + ".xfm")
            return candidate if candidate.is_file() else None
    return None
