"""volumes.tsv columns, native volumes and the missing-transform warning, without the network."""
import argparse
import csv
import json
import logging
import math

import numpy as np
import pytest

pytest.importorskip("minc2_simple")
from hvr_cnn import minctools, pipeline, volumes  # noqa: E402
from hvr_cnn.inputs import Scan  # noqa: E402
from test_geometry import make_minc  # noqa: E402

SIMPLE = {11: 40, 12: 10, 21: 30, 22: 10}
needs_minc_tools = pytest.mark.skipif(minctools.which("param2xfm") is None, reason="MINC tools not installed")


def read_tsv(path):
    with open(str(path), newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def ident(scan_id, status):
    return dict(id=scan_id, subject="", session="", group="", model="simple", input_space="stx", status=status)


def ok_row(scan_id, scale):
    return dict(ident(scan_id, "ok"), **volumes.add_native(volumes.summarise(SIMPLE, "simple"), scale))


def test_failed_first_scan_keeps_every_column(tmp_path):
    pipeline.write_volumes(tmp_path / "v.tsv", [ident("a", "failed"), ok_row("b", 0.9)])
    header, rows = read_tsv(tmp_path / "v.tsv")
    assert header == list(pipeline.ID_COLUMNS) + volumes.columns("simple")
    assert rows[0]["L_HC_mm3_stx"] == "" and rows[1]["L_HC_mm3_stx"] == "40"
    assert float(rows[1]["L_HC_mm3_native"]) == pytest.approx(40 / 0.9)


def test_scan_without_transform_first_keeps_native_columns_of_the_others(tmp_path):
    pipeline.write_volumes(tmp_path / "v.tsv", [ok_row("a", None), ok_row("b", 0.5)])
    _, rows = read_tsv(tmp_path / "v.tsv")
    assert rows[0]["L_HC_mm3_native"] == "" and rows[0]["scale_factor"] == ""
    assert rows[1]["L_HC_mm3_native"] == "80" and rows[1]["scale_factor"] == "0.5"


def finished_scan(tmp_path, xfm=None):
    """A stereotaxic input whose label file already exists, so process_scan
    takes the `skipped` path and needs no network."""
    inp = tmp_path / "in" / "stx2_s1_t1.mnc"
    inp.parent.mkdir(parents=True)
    inp.touch()
    seg = tmp_path / "out" / "s1" / "s1_space-stx_model-simple_seg.mnc"
    seg.parent.mkdir(parents=True)
    data = np.zeros((20, 20, 20), np.float32)
    for i, lab in enumerate(SIMPLE):  # 4 x 4 x 4 = 64 voxels per label
        data[i * 4:(i + 1) * 4, 2:6, 2:6] = lab
    make_minc(seg, (20, 20, 20), (-96.0, -132.0, -78.0), data=data)
    args = argparse.Namespace(out_format="auto", model="simple", overwrite=False, input_space="stx")
    return Scan("s1", inp, xfm=xfm), args, tmp_path / "out"


def scaling_xfm(path, scales=("1.1", "1.0", "0.9")):
    path.parent.mkdir(parents=True, exist_ok=True)
    minctools.run(["param2xfm", "-clobber", "-scales"] + list(scales) + [path])
    return path


def process(scan, args, outdir):
    return pipeline.process_scan(scan, args, None, "cpu", outdir, outdir / "work")


def test_no_transform_is_a_warning_and_empty_native_cells(tmp_path, caplog):
    scan, args, outdir = finished_scan(tmp_path)
    with caplog.at_level(logging.WARNING):
        row, info = process(scan, args, outdir)
    assert info["status"] == "skipped"
    assert info["native_volumes"] == "not computed: no transform"
    assert math.isnan(row["scale_factor"]) and math.isnan(row["L_HC_mm3_native"])
    assert row["L_HC_mm3_stx"] == 64
    assert any("native volumes not computed: no transform" in m for m in caplog.messages)
    assert any("native volumes not computed: no transform" in w for w in info["warnings"])


@needs_minc_tools
def test_xfm_column_gives_native_volumes(tmp_path):
    xfm = scaling_xfm(tmp_path / "elsewhere" / "s1.xfm")
    scan, args, outdir = finished_scan(tmp_path, xfm=xfm)
    row, info = process(scan, args, outdir)
    assert info["native_volumes"] == "computed" and info["scale_xfm"] == str(xfm)
    assert row["scale_factor"] == pytest.approx(1.1 * 1.0 * 0.9, rel=1e-6)
    assert row["L_HC_mm3_native"] == pytest.approx(64 / 0.99, rel=1e-6)
    assert "warnings" not in info


@needs_minc_tools
def test_pipeline_xfm_next_to_the_input_gives_native_volumes(tmp_path):
    scan, args, outdir = finished_scan(tmp_path)
    scaling_xfm(scan.path.with_name("stx2_s1_t1.xfm"), scales=("1.2", "1.2", "1.2"))
    row, info = process(scan, args, outdir)
    assert row["scale_factor"] == pytest.approx(1.728, rel=1e-6)
    assert info["native_volumes"] == "computed"


def test_run_table_input_space_and_one_summary_warning(tmp_path, monkeypatch, caplog):
    """run() end to end with the segmentation replaced: a failed first scan, a
    scan without a transform, a native one."""
    def fake_process(scan, args, model, device, outdir, work_root):
        if scan.id == "bad":
            raise pipeline.ScanError("boom")
        native = scan.id == "raw"
        row = volumes.add_native(volumes.summarise(SIMPLE, "simple"), 0.9 if native else None)
        return row, {"id": scan.id, "input": str(scan.path), "status": "ok", "outputs": [],
                     "input_space": "native" if native else "stx",
                     "native_volumes": "computed" if native else volumes.NO_TRANSFORM}

    monkeypatch.setattr(pipeline, "process_scan", fake_process)
    monkeypatch.setattr(pipeline, "load_model", lambda model, device: None)
    args = argparse.Namespace(output=str(tmp_path / "out"), threads=1, device="cpu", model="simple",
                              input_space="auto", work_dir=None, keep_work=False, fail_fast=False)
    scans = [Scan(i, tmp_path / (i + ".mnc")) for i in ("bad", "stx", "raw")]
    with caplog.at_level(logging.WARNING):
        assert pipeline.run(args, scans) == 1
    header, rows = read_tsv(tmp_path / "out" / "volumes.tsv")
    assert header == list(pipeline.ID_COLUMNS) + volumes.columns("simple")
    assert [r["input_space"] for r in rows] == ["auto", "stx", "native"]  # the decided space, not always stx
    assert rows[1]["L_HC_mm3_native"] == "" and float(rows[2]["L_HC_mm3_native"]) == pytest.approx(40 / 0.9)
    summary = [m for m in caplog.messages if "scan(s) with results" in m]
    assert len(summary) == 1 and "for 1 of 2 scan(s)" in summary[0]
    record = json.load(open(str(tmp_path / "out" / "run.json")))
    assert [s.get("native_volumes") for s in record["scans"]] == [None, "not computed: no transform", "computed"]
