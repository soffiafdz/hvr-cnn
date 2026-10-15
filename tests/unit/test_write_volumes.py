"""`volumes.tsv` as written by the pipeline."""
import csv

import pytest

pytest.importorskip("minc2_simple")
from hvr_cnn import pipeline  # noqa: E402


def ident(sid, status):
    return dict(id=sid, subject="", session="", group="", model="simple", input_space="stx", status=status)


def ok_row(sid):
    return dict(ident(sid, "ok"), L_HC_mm3=3000, L_VC_mm3=500, L_HVR=0.857, missing_labels=0)


def read(path):
    with open(str(path), newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_failed_first_scan_keeps_the_columns(tmp_path):
    path = tmp_path / "volumes.tsv"
    pipeline.write_volumes(path, [ident("a", "failed"), ok_row("b")])
    rows = read(path)
    assert [r["status"] for r in rows] == ["failed", "ok"]
    assert rows[0]["L_HVR"] == "" and rows[1]["L_HVR"] == "0.857"


def test_column_order(tmp_path):
    path = tmp_path / "volumes.tsv"
    pipeline.write_volumes(path, [ok_row("a"), ident("b", "failed")])
    header = path.read_text().splitlines()[0].split("\t")
    assert header == list(pipeline.ID_COLUMNS) + ["L_HC_mm3", "L_VC_mm3", "L_HVR", "missing_labels"]


def test_unix_line_endings(tmp_path):
    path = tmp_path / "volumes.tsv"
    pipeline.write_volumes(path, [ok_row("a"), ident("b", "failed")])
    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b"\tfailed" + b"\t" * 4 + b"\n")
