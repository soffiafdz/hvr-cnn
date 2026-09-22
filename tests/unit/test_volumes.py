import math

import numpy as np
import pytest

from hvr_cnn import volumes

# Voxel counts produced by the previous implementation on one test scan.
BASELINE_SIMPLE = {11: 4716, 12: 1477, 21: 4507, 22: 1436}
BASELINE_DETAILED = {
    111: 772, 112: 1775, 113: 2253, 121: 228, 122: 577, 123: 608, 130: 1028,
    211: 734, 212: 1539, 213: 2349, 221: 244, 222: 603, 223: 538, 230: 1219,
}


def test_expected_labels():
    assert volumes.expected_labels("simple") == {11, 12, 21, 22}
    assert volumes.expected_labels("detailed") == set(BASELINE_DETAILED)


def test_label_sets_do_not_overlap():
    assert not volumes.expected_labels("simple") & volumes.expected_labels("detailed")


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        volumes.expected_labels("extra")


def test_count_labels_ignores_background():
    seg = np.zeros((4, 4, 4), dtype=np.uint8)
    seg[0, 0, :3] = 11
    seg[1, 1, 1] = 22
    assert volumes.count_labels(seg) == {11: 3, 22: 1}


def test_count_labels_accepts_integral_floats():
    # minc2_simple hands labels back as float64
    assert volumes.count_labels(np.array([0.0, 111.0, 111.0])) == {111: 2}


def test_count_labels_rejects_non_integer():
    with pytest.raises(ValueError, match="non-integer"):
        volumes.count_labels(np.array([0.0, 11.5]))


def test_simple_matches_old_code():
    row = volumes.summarise(BASELINE_SIMPLE, "simple")
    assert (row["L_HC_vox"], row["L_VC_vox"]) == (4716, 1477)
    assert (row["R_HC_vox"], row["R_VC_vox"]) == (4507, 1436)
    # values written by the old container for the same segmentation
    assert row["L_HVR"] == pytest.approx(0.7615049249152269, abs=1e-15)
    assert row["R_HVR"] == pytest.approx(0.7583711930001683, abs=1e-15)
    assert row["missing_labels"] == 0


def test_detailed_sums_sublabels():
    # the case the old container crashed on
    row = volumes.summarise(BASELINE_DETAILED, "detailed")
    assert (row["L_HC_vox"], row["L_VC_vox"], row["L_AMY_vox"]) == (4800, 1413, 1028)
    assert (row["R_HC_vox"], row["R_VC_vox"], row["R_AMY_vox"]) == (4622, 1385, 1219)
    assert row["L_HVR"] == pytest.approx(4800 / 6213)
    assert row["R_HVR"] == pytest.approx(4622 / 6007)
    # x11 = tail (posterior), x13 = head (anterior, next to the amygdala)
    assert (row["L_HC_tail_vox"], row["L_HC_body_vox"], row["L_HC_head_vox"]) == (772, 1775, 2253)
    assert (row["L_VC_tail_vox"], row["L_VC_body_vox"], row["L_VC_head_vox"]) == (228, 577, 608)
    assert "L_AMY_head_vox" not in row


def test_amygdala_not_in_hvr():
    with_amy = volumes.summarise(BASELINE_DETAILED, "detailed")
    without = volumes.summarise(
        {k: v for k, v in BASELINE_DETAILED.items() if k not in (130, 230)}, "detailed"
    )
    assert with_amy["L_HVR"] == without["L_HVR"]
    assert without["missing_labels"] == 2


def test_wrong_model_is_an_error_not_zero_volumes():
    with pytest.raises(ValueError, match="do not belong to model 'simple'"):
        volumes.summarise(BASELINE_DETAILED, "simple")


def test_mm3_only_when_it_differs_from_vox():
    row = volumes.summarise(BASELINE_SIMPLE, "simple", voxel_volume_mm3=0.5)
    assert row["L_HC_mm3"] == 2358.0
    assert row["L_HC_vox"] == 4716
    unit = volumes.summarise(BASELINE_SIMPLE, "simple")
    assert not any(k.endswith("_mm3") for k in unit)
    assert len(unit) == 7  # 2 sides x (HC, VC, HVR) + missing_labels


def test_empty_segmentation_gives_nan_not_crash():
    row = volumes.summarise({}, "simple")
    assert row["L_HC_vox"] == 0
    assert math.isnan(row["L_HVR"]) and math.isnan(row["R_HVR"])
    assert row["missing_labels"] == 4


def test_hvr_edges():
    assert volumes.hvr(1, 0) == 1.0
    assert volumes.hvr(0, 1) == 0.0
    assert math.isnan(volumes.hvr(0, 0))
    with pytest.raises(ValueError):
        volumes.hvr(-1, 2)
