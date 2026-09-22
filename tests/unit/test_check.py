"""`hvr-cnn check` on a synthetic run directory."""
import json

import numpy as np
import pytest

pytest.importorskip("minc2_simple")
from hvr_cnn import check, pipeline, volumes  # noqa: E402
from test_geometry import make_minc  # noqa: E402


def fake_run(outdir, model="simple", labels=(11, 12, 21, 22), hvr=(0.7, 0.7), status="ok"):
    """A run directory with one scan whose label file holds `labels`."""
    scan_dir = outdir / "s1"
    scan_dir.mkdir(parents=True)
    seg = scan_dir / ("s1_space-stx_model-%s_seg.mnc" % model)
    data = np.zeros((20, 20, 20), np.float32)
    for i, lab in enumerate(labels):
        data[i * 4:(i + 1) * 4, 2:6, 2:6] = lab
    make_minc(seg, (20, 20, 20), (-96.0, -132.0, -78.0), data=data)
    counts = volumes.count_labels(data)
    row = dict(id="s1", subject="", session="", group="", model=model, input_space="stx", status=status)
    if status == "ok":
        row.update(volumes.summarise(counts, model) if set(counts) <= volumes.expected_labels(model) else {})
        row["L_HVR"], row["R_HVR"] = hvr
    pipeline.write_volumes(outdir / "volumes.tsv", [row])
    json.dump({"arguments": {"model": model},
               "scans": [{"id": "s1", "input": "/nowhere/s1.mnc", "status": status, "outputs": [str(seg)]}]},
              open(str(outdir / "run.json"), "w"))
    return outdir


def results(outdir, reference=None):
    return list(check.check_run(outdir, reference))


def failures(outdir, reference=None):
    return [m for ok, m in results(outdir, reference) if not ok]


def test_good_run_passes(tmp_path):
    fake_run(tmp_path / "a")
    assert failures(tmp_path / "a") == []


def test_missing_label_detected(tmp_path):
    fake_run(tmp_path / "a", labels=(11, 12, 21))
    assert any("label set wrong" in m for m in failures(tmp_path / "a"))


def test_hvr_out_of_range_detected(tmp_path):
    fake_run(tmp_path / "a", hvr=(1.0, 0.7))
    bad = failures(tmp_path / "a")
    assert len(bad) == 1 and "L_HVR" in bad[0]


def test_failed_scan_is_reported_not_checked(tmp_path):
    fake_run(tmp_path / "a", status="failed")
    msgs = failures(tmp_path / "a")
    assert msgs == ["0 scan(s) with results"]


def test_reference_identical_passes(tmp_path):
    fake_run(tmp_path / "a")
    fake_run(tmp_path / "b")
    assert failures(tmp_path / "a", tmp_path / "b") == []
    assert any("Dice vs reference min 1.0000" in m for _, m in results(tmp_path / "a", tmp_path / "b"))


def test_reference_with_other_model_fails(tmp_path):
    fake_run(tmp_path / "a")
    fake_run(tmp_path / "b", model="detailed", labels=(111, 112, 113, 121, 122, 123, 130,
                                                       211, 212, 213, 221, 222, 223, 230))
    assert any("reference model detailed vs simple" in m for m in failures(tmp_path / "a", tmp_path / "b"))


def test_missing_run_json(tmp_path):
    assert any("cannot read" in m for m in failures(tmp_path))
