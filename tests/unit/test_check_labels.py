"""Which incomplete segmentations fail a scan."""
import pytest

pytest.importorskip("minc2_simple")
from hvr_cnn import pipeline, volumes  # noqa: E402

FULL = {v: 100 for v in volumes.expected_labels("detailed")}


def test_complete_segmentation_passes():
    assert pipeline.check_labels(FULL, "detailed", "seg.mnc") is None


def test_missing_part_is_a_warning():
    counts = {k: v for k, v in FULL.items() if k != 111}
    note = pipeline.check_labels(counts, "detailed", "seg.mnc")
    assert note.startswith("labels 111 absent")


def test_empty_structure_fails():
    counts = {k: v for k, v in FULL.items() if k not in (121, 122, 123)}
    with pytest.raises(pipeline.ScanError, match="L_VC empty"):
        pipeline.check_labels(counts, "detailed", "seg.mnc")


def test_simple_missing_label_fails():
    with pytest.raises(pipeline.ScanError, match="R_VC empty"):
        pipeline.check_labels({11: 1, 12: 1, 21: 1}, "simple", "seg.mnc")
