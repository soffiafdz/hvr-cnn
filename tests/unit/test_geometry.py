"""Geometry guard and input-space decision on synthetic MINC headers."""
import numpy as np
import pytest

minc2_simple = pytest.importorskip("minc2_simple")
from hvr_cnn import segment  # noqa: E402


def make_minc(path, lengths, starts, steps=(1.0, 1.0, 1.0), value=60.0, data=None):
    from minc2_simple import minc2_file, minc2_dim

    dims = [minc2_dim(id=i + 1, length=n, start=s0, step=st, have_dir_cos=False, dir_cos=None)
            for i, (n, s0, st) in enumerate(zip(lengths, starts, steps))]
    f = minc2_file()
    f.define(dims, minc2_file.MINC2_FLOAT, minc2_file.MINC2_FLOAT)
    f.create(str(path))
    f.setup_standard_order()
    if data is None:
        data = np.full((lengths[2], lengths[1], lengths[0]), value, dtype=np.float32)
    f.save_complete_volume(np.ascontiguousarray(data, dtype=np.float32))
    f.close()
    return path


ICBM = dict(lengths=(193, 229, 193), starts=(-96.0, -132.0, -78.0))


def test_icbm_grid_is_stx(tmp_path):
    p = make_minc(tmp_path / "a.mnc", **ICBM)
    assert segment.decide_space(p)[0] == "stx"
    segment.check_stx_geometry(p)


def test_assemblynet_grid_is_stx_aligned(tmp_path):
    p = make_minc(tmp_path / "a.mnc", lengths=(181, 217, 181), starts=(-90.0, -126.0, -72.0))
    assert segment.decide_space(p)[0] == "stx"


def test_native_origin_is_not_stx(tmp_path):
    p = make_minc(tmp_path / "a.mnc", lengths=(208, 240, 256), starts=(-94.729, -113.629, -142.391))
    space, reason = segment.decide_space(p)
    assert space == "native" and "not aligned" in reason


def test_wrong_voxel_size_rejected(tmp_path):
    p = make_minc(tmp_path / "a.mnc", lengths=(96, 114, 96), starts=(-96.0, -132.0, -78.0), steps=(2.0, 2.0, 2.0))
    assert segment.decide_space(p)[0] == "native"
    with pytest.raises(segment.GeometryError, match="voxel size"):
        segment.check_stx_geometry(p)


def test_field_of_view_too_small_rejected(tmp_path):
    p = make_minc(tmp_path / "a.mnc", lengths=(100, 229, 193), starts=(-96.0, -132.0, -78.0))
    with pytest.raises(segment.GeometryError, match="x covers"):
        segment.check_stx_geometry(p)


def test_intensity_p90(tmp_path):
    p = make_minc(tmp_path / "a.mnc", value=80.0, **ICBM)
    assert segment.intensity_p90(p) == pytest.approx(80.0)
    lo, hi = segment.INTENSITY_P90_RANGE
    assert lo < 80.0 < hi
