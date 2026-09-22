import pytest

from hvr_cnn import inputs
from hvr_cnn.inputs import InputError


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.mark.parametrize("name,expected", [
    ("stx2_s01_t1.mnc", "stx2_s01_t1"),
    ("sub-01_ses-02_T1w.nii.gz", "sub-01_ses-02_T1w"),
    ("a.b.NII", "a.b"),
    ("scan.mnc.gz", "scan"),
])
def test_strip_extension(name, expected):
    assert inputs.strip_extension(name) == expected


@pytest.mark.parametrize("name", ["scan.dcm", "scan.img", "scan", "scan.nii.bz2"])
def test_unsupported_extension(name):
    with pytest.raises(InputError, match="unsupported file type"):
        inputs.strip_extension(name)


def test_image_format():
    assert inputs.image_format("a/b/x.mnc.gz") == "mnc"
    assert inputs.image_format("x.nii.gz") == "nii"


def test_scan_id_from_subject_and_session():
    assert inputs.scan_id("whatever.mnc", "003 S/1", "2021-08-23") == "003-S-1_2021-08-23"
    assert inputs.scan_id("whatever.mnc", "s1") == "s1"
    assert inputs.scan_id("dir/my scan.nii") == "my-scan"


def test_from_paths(tmp_path):
    a, b = touch(tmp_path / "a.mnc"), touch(tmp_path / "b.nii.gz")
    scans = inputs.from_paths([str(a), str(b)])
    assert [s.id for s in scans] == ["a", "b"]
    assert scans[0].subject is None


def test_all_problems_reported_at_once(tmp_path):
    ok = touch(tmp_path / "ok.mnc")
    with pytest.raises(InputError) as exc:
        inputs.from_paths([str(ok), str(tmp_path / "missing.mnc"), str(tmp_path / "x.dcm")])
    text = str(exc.value)
    assert "2 problem(s)" in text and "missing.mnc: no such file" in text and "x.dcm" in text


def test_duplicate_ids_rejected(tmp_path):
    a, b = touch(tmp_path / "v1" / "t1.mnc"), touch(tmp_path / "v2" / "t1.mnc")
    with pytest.raises(InputError, match="identifier 't1' occurs 2 times"):
        inputs.from_paths([str(a), str(b)])


def test_csv_relative_paths_and_optional_columns(tmp_path):
    touch(tmp_path / "data" / "s1.mnc")
    touch(tmp_path / "data" / "s2.nii.gz")
    table = tmp_path / "scans.csv"
    table.write_text("subject,session,input,group\n"
                     "s1,bl,data/s1.mnc,CN\n"
                     "s2,,data/s2.nii.gz,\n")
    scans = inputs.from_csv(table)
    assert [s.id for s in scans] == ["s1_bl", "s2"]
    assert scans[0].path == tmp_path / "data" / "s1.mnc"
    assert (scans[0].group, scans[1].group, scans[1].session) == ("CN", None, None)


def test_csv_same_file_name_disambiguated_by_subject(tmp_path):
    touch(tmp_path / "a" / "t1.mnc")
    touch(tmp_path / "b" / "t1.mnc")
    table = tmp_path / "scans.tsv"
    table.write_text("input\tsubject\na/t1.mnc\ta\nb/t1.mnc\tb\n")
    assert [s.id for s in inputs.from_csv(table)] == ["a", "b"]


def test_csv_with_bom_and_semicolons(tmp_path):
    touch(tmp_path / "s1.mnc")
    table = tmp_path / "excel.csv"
    table.write_bytes("﻿input;subject\ns1.mnc;s1\n".encode("utf-8"))
    assert inputs.from_csv(table)[0].id == "s1"


def test_csv_without_header_is_explained(tmp_path):
    table = tmp_path / "old.csv"
    table.write_text("s1,bl,/data/s1.mnc\n")
    with pytest.raises(InputError, match="needs a header row with a column named 'input'"):
        inputs.from_csv(table)


def test_csv_empty_input_cell_names_the_line(tmp_path):
    table = tmp_path / "scans.csv"
    table.write_text("input,subject\n,s1\n")
    with pytest.raises(InputError, match="line 2: empty 'input'"):
        inputs.from_csv(table)


def test_csv_missing_file(tmp_path):
    with pytest.raises(InputError, match="no such file"):
        inputs.from_csv(tmp_path / "nope.csv")


def test_csv_extension_checked_even_with_subject(tmp_path):
    touch(tmp_path / "s1.dcm")
    table = tmp_path / "scans.csv"
    table.write_text("input,subject\ns1.dcm,s1\n")
    with pytest.raises(InputError, match="unsupported file type"):
        inputs.from_csv(table)


def test_assemblynet_pair_detection(tmp_path):
    from hvr_cnn import preprocess

    t1 = touch(tmp_path / "mni_t1_sub-01_T1w.nii.gz")
    assert preprocess.assemblynet_mask_for(t1) is None  # no mask yet
    mask = touch(tmp_path / "mni_mask_sub-01_T1w.nii.gz")
    assert preprocess.assemblynet_mask_for(t1) == mask
    assert preprocess.assemblynet_mask_for(touch(tmp_path / "native_t1_sub-01_T1w.nii.gz")) is None
    assert preprocess.assemblynet_mask_for(touch(tmp_path / "stx2_sub-01_t1.mnc")) is None
