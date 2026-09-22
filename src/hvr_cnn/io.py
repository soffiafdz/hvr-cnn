"""The NIfTI boundary: NIfTI -> MINC for processing, MINC labels -> NIfTI on
the grid of the input.

Both directions are done here with nibabel and minc2_simple rather than
with `itk_convert`, which mirrors the volume for NIfTI files whose array
axes do not run R->L... A->P... I->S (verified on an AssemblyNet
`native_t1`) and writes a left-right mirrored affine when going MINC ->
NIfTI.

Conventions: MINC world coordinates are RAS, like NIfTI's. A MINC volume in
"standard order" is stored z, y, x with positive steps; world =
sum_i dir_cos_i * (start_i + step_i * index_i).
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class ConversionError(ValueError):
    pass


def _nifti_image(path):
    import nibabel as nib

    img = nib.load(str(path))
    if img.ndim == 4 and img.shape[3] == 1:
        img = img.slicer[..., 0]
    if img.ndim != 3:
        raise ConversionError("%s: expected a 3D image, got shape %s" % (path, img.shape))
    return img


def _decompose(affine):
    """Affine -> (direction cosines as columns, steps, starts) for a MINC grid.
    Raises for sheared / non-orthogonal axes."""
    m = np.asarray(affine, dtype=float)[:3, :3]
    steps = np.linalg.norm(m, axis=0)
    if np.any(steps <= 0):
        raise ConversionError("degenerate affine (zero voxel size)")
    cos = m / steps
    if not np.allclose(cos.T @ cos, np.eye(3), atol=1e-4):
        raise ConversionError("sheared or non-orthogonal voxel axes are not supported")
    starts = cos.T @ np.asarray(affine, dtype=float)[:3, 3]
    return cos, steps, starts


def _compose(cos, steps, starts):
    affine = np.eye(4)
    affine[:3, :3] = cos * steps
    affine[:3, 3] = cos @ starts
    return affine


def nifti_to_minc(nii_path, mnc_path):
    """Write the NIfTI as a float MINC volume with the same grid.

    The array is reordered to RAS-closest axis order first (nibabel's
    `as_closest_canonical`), so the MINC has positive steps; oblique grids
    keep their direction cosines. nibabel's affine choice applies: sform
    when its code is set, else qform.
    """
    import nibabel as nib
    from minc2_simple import minc2_dim, minc2_file

    img = nib.as_closest_canonical(_nifti_image(nii_path))
    return write_minc(np.asarray(img.dataobj, dtype=np.float32), img.affine, mnc_path)


def write_minc(data, affine, mnc_path, decimals=None):
    """Write `data` (x, y, z index order) with a RAS `affine` as a float MINC
    volume in z, y, x storage order with positive steps. With `decimals`,
    starts and steps are rounded and near-identity direction cosines snapped
    to exact ones, so the same grid gives the same header whatever wrote the
    source file (the registration is sensitive to that)."""
    from minc2_simple import minc2_dim, minc2_file

    cos, steps, starts = _decompose(affine)
    if decimals is not None:
        if np.allclose(cos, np.eye(3), atol=1e-6):
            cos = np.eye(3)
        starts = np.round(starts, decimals)
        steps = np.round(steps, decimals + 2)
    dims = [
        minc2_dim(id=i + 1, length=int(data.shape[i]), start=float(starts[i]), step=float(steps[i]),
                  have_dir_cos=True, dir_cos=np.ascontiguousarray(cos[:, i], dtype=np.float64))
        for i in range(3)
    ]
    out = minc2_file()
    out.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
    out.create(str(mnc_path))
    out.setup_standard_order()
    out.save_complete_volume(np.ascontiguousarray(np.asarray(data, dtype=np.float32).transpose(2, 1, 0)))
    out.close()
    return mnc_path


def read_minc(path, dtype=np.float64):
    """(array in x, y, z index order, RAS affine) of a MINC volume."""
    from minc2_simple import minc2_file

    f = minc2_file(str(path))
    f.setup_standard_order()
    data = np.asarray(f.load_complete_volume(np.dtype(dtype).name)).transpose(2, 1, 0)
    dims = f.representation_dims()
    f.close()
    cos = np.eye(3)
    for i, d in enumerate(dims):
        if getattr(d, "have_dir_cos", 0) and d.dir_cos is not None:
            cos[:, i] = np.asarray(d.dir_cos, dtype=float)
    affine = _compose(cos, np.array([d.step for d in dims], float), np.array([d.start for d in dims], float))
    return data, affine


def minc_labels_to_nifti(mnc_path, out_path, like=None, description="hvr-cnn labels"):
    """Write MINC labels as NIfTI. With `like` (a NIfTI path) the output gets
    exactly that file's array axis order, shape, affine and q/s-form codes,
    after checking that the two grids coincide; otherwise RAS order."""
    import nibabel as nib
    from nibabel import orientations as ori

    data, affine = read_minc(mnc_path)
    if not np.array_equal(data, np.rint(data)):
        raise ConversionError("%s: not a label volume" % mnc_path)
    dtype = np.uint8 if data.max() < 256 else np.uint16
    img = nib.Nifti1Image(data.astype(dtype), affine)
    if like is not None:
        ref = _nifti_image(like)
        transform = ori.ornt_transform(ori.io_orientation(img.affine), ori.io_orientation(ref.affine))
        img = img.as_reoriented(transform)
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ConversionError(
                "label grid %s / %s does not match the input's %s / %s"
                % (img.shape, np.round(img.affine, 3).tolist(), ref.shape, np.round(ref.affine, 3).tolist()))
        header = ref.header.copy()
        header.set_data_dtype(dtype)
        out = nib.Nifti1Image(np.asarray(img.dataobj), ref.affine, header)
        out.set_qform(ref.get_qform(), int(ref.header["qform_code"]))
        out.set_sform(ref.get_sform(), int(ref.header["sform_code"]))
    else:
        out = img
    out.header.set_slope_inter(1.0, 0.0)
    out.header["cal_min"] = 0
    out.header["cal_max"] = 0
    out.header["descrip"] = description.encode()[:79]
    out.header["intent_code"] = 0
    out.set_data_dtype(dtype)
    nib.save(out, str(out_path))
    return out_path
