"""NIfTI <-> MINC round trips on synthetic volumes in several orientations."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("minc2_simple")
from hvr_cnn import io  # noqa: E402

SHAPE = (12, 10, 8)


def synthetic(tmp_path, name, affine, labels=False):
    rng = np.random.default_rng(0)
    if labels:
        data = rng.integers(0, 3, SHAPE).astype(np.uint8) * 11
    else:
        data = rng.normal(50, 10, SHAPE).astype(np.float32)
    path = tmp_path / name
    nib.Nifti1Image(data, affine).to_filename(str(path))
    return path, data


AFFINES = {
    "RAS": np.diag([1.0, 1.0, 1.0, 1.0]),
    "LPI": np.array([[-1, 0, 0, 5.0], [0, -1, 0, 7.0], [0, 0, -1, 9.0], [0, 0, 0, 1]]),
    "LAS_0.7mm": np.array([[-0.7, 0, 0, 3.0], [0, 0.7, 0, -4.0], [0, 0, 0.7, 2.0], [0, 0, 0, 1]]),
    "permuted": np.array([[0, 0, 1, 1.0], [1, 0, 0, 2.0], [0, 1, 0, 3.0], [0, 0, 0, 1]]),
}
theta = np.deg2rad(7)
AFFINES["oblique"] = np.array([[np.cos(theta), -np.sin(theta), 0, -10.0],
                               [np.sin(theta), np.cos(theta), 0, -20.0],
                               [0, 0, 1, -30.0], [0, 0, 0, 1]])


@pytest.mark.parametrize("name", sorted(AFFINES))
def test_nifti_to_minc_keeps_world_coordinates(tmp_path, name):
    path, data = synthetic(tmp_path, "in.nii.gz", AFFINES[name])
    mnc = io.nifti_to_minc(path, tmp_path / "in.mnc")
    arr, affine = io.read_minc(mnc, np.float32)
    # the MINC, read back in RAS index order, must equal nibabel's canonical view
    can = nib.as_closest_canonical(nib.load(str(path)))
    assert arr.shape == can.shape
    assert np.allclose(affine, can.affine, atol=1e-5)
    assert np.allclose(arr, np.asarray(can.dataobj), atol=1e-6)


@pytest.mark.parametrize("name", sorted(AFFINES))
def test_labels_back_on_the_input_grid(tmp_path, name):
    path, data = synthetic(tmp_path, "in.nii.gz", AFFINES[name], labels=True)
    mnc = io.nifti_to_minc(path, tmp_path / "in.mnc")
    out = io.minc_labels_to_nifti(mnc, tmp_path / "out.nii.gz", like=path)
    ref, res = nib.load(str(path)), nib.load(str(out))
    assert res.shape == ref.shape
    assert np.allclose(res.affine, ref.affine, atol=1e-5)
    assert np.array_equal(np.asarray(res.dataobj), data)  # same array order as the input
    assert res.get_data_dtype() == np.uint8
    assert nib.aff2axcodes(res.affine) == nib.aff2axcodes(ref.affine)


def test_labels_without_like_are_ras(tmp_path):
    path, data = synthetic(tmp_path, "in.nii.gz", AFFINES["LPI"], labels=True)
    mnc = io.nifti_to_minc(path, tmp_path / "in.mnc")
    out = nib.load(str(io.minc_labels_to_nifti(mnc, tmp_path / "out.nii.gz")))
    assert nib.aff2axcodes(out.affine) == ("R", "A", "S")
    can = nib.as_closest_canonical(nib.load(str(path)))
    assert np.array_equal(np.asarray(out.dataobj), np.asarray(can.dataobj))


def test_mismatched_like_is_an_error(tmp_path):
    path, _ = synthetic(tmp_path, "in.nii.gz", AFFINES["RAS"], labels=True)
    other, _ = synthetic(tmp_path, "other.nii.gz", AFFINES["LAS_0.7mm"], labels=True)
    mnc = io.nifti_to_minc(path, tmp_path / "in.mnc")
    with pytest.raises(io.ConversionError, match="does not match"):
        io.minc_labels_to_nifti(mnc, tmp_path / "out.nii.gz", like=other)


def test_sheared_affine_rejected(tmp_path):
    affine = np.eye(4)
    affine[0, 1] = 0.3
    path, _ = synthetic(tmp_path, "in.nii.gz", affine)
    with pytest.raises(io.ConversionError, match="sheared"):
        io.nifti_to_minc(path, tmp_path / "in.mnc")


def test_4d_singleton_accepted_and_real_4d_rejected(tmp_path):
    data = np.zeros(SHAPE + (1,), np.float32)
    p = tmp_path / "a.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(str(p))
    io.nifti_to_minc(p, tmp_path / "a.mnc")
    nib.Nifti1Image(np.zeros(SHAPE + (2,), np.float32), np.eye(4)).to_filename(str(p))
    with pytest.raises(io.ConversionError, match="3D"):
        io.nifti_to_minc(p, tmp_path / "b.mnc")
