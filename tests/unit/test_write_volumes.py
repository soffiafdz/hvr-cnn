"""`volumes.tsv` line endings. Its columns are tested in test_pipeline.py."""
import pytest

pytest.importorskip("minc2_simple")
from hvr_cnn import pipeline, volumes  # noqa: E402


def ident(sid, status):
    return dict(id=sid, subject="", session="", group="", model="simple", input_space="stx", status=status)


def test_unix_line_endings(tmp_path):
    path = tmp_path / "volumes.tsv"
    ok = dict(ident("a", "ok"), **volumes.add_native(volumes.summarise({11: 40, 12: 10, 21: 30, 22: 10}, "simple"), 0.9))
    pipeline.write_volumes(path, [ok, ident("b", "failed")])
    data = path.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    lines = data.decode().split("\n")[:-1]
    assert len(lines) == 3 and len({line.count("\t") for line in lines}) == 1  # header and both rows, same width
