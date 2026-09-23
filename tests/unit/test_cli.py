import os
import sys

import pytest

from hvr_cnn import __version__, cli, minctools


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "hvr-cnn " + __version__


def test_no_arguments_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "commands" in capsys.readouterr().err


def test_run_requires_output():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "-i", "t1.mnc"])
    assert exc.value.code == 2


def test_help_shows_examples_and_exit_status(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--help"])
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "examples:" in out and "exit status:" in out and "--dry-run" in out


def test_verbose_and_quiet_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["selftest", "-v", "-q"])


def test_threads_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["run", "-i", "a.mnc", "-o", "out", "--threads", "0"])


def test_default_threads_honours_slurm(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert cli.default_threads() == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "100000")
    assert 1 <= cli.default_threads() <= (os.cpu_count() or 1)


def test_legacy_command_line_gets_migration_hint(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["-f", "/app/data/input.csv", "-o", "/app/output", "--qc_images", "--hc_vols"])
    err = capsys.readouterr().err
    assert exc.value.code == cli.EXIT_USAGE
    assert "0.0.x" in err and "--csv" in err and "volumes.tsv" in err


def test_new_command_line_is_not_mistaken_for_legacy():
    assert cli.legacy_hint(["run", "-i", "a.mnc", "-o", "out"]) is None
    assert cli.legacy_hint(["--version"]) is None


def test_missing_input_is_usage_error_not_traceback(tmp_path, caplog):
    code = cli.main(["run", "-i", str(tmp_path / "nope.mnc"), "-o", str(tmp_path / "out")])
    assert code == cli.EXIT_USAGE
    assert "no such file" in caplog.text
    assert not (tmp_path / "out").exists()


def test_dry_run_lists_plan_and_writes_nothing(tmp_path, caplog):
    scan = tmp_path / "sub-01_T1w.nii.gz"
    scan.write_bytes(b"")
    caplog.set_level("INFO")
    code = cli.main(["run", "-i", str(scan), "-o", str(tmp_path / "out"), "--dry-run"])
    assert code == cli.EXIT_OK
    assert "sub-01_T1w" in caplog.text and "labels as nii" in caplog.text
    assert not (tmp_path / "out").exists()


def test_run_input_and_csv_exclusive():
    with pytest.raises(SystemExit):
        cli.main(["run", "-i", "t1.mnc", "--csv", "a.csv", "-o", "out"])


def test_run_defaults():
    args = cli.build_parser().parse_args(["run", "-i", "a.mnc", "b.nii.gz", "-o", "out"])
    assert args.input == ["a.mnc", "b.nii.gz"]
    assert args.qc is True
    assert args.denoise is True and args.field == 3.0
    off = cli.build_parser().parse_args(["run", "-i", "a.mnc", "-o", "out", "--no-denoise", "--field", "1.5"])
    assert off.denoise is False and off.field == 1.5
    assert cli.build_parser().parse_args(["run", "-i", "a.mnc", "-o", "out", "--no-qc"]).qc is False
    assert (args.model, args.input_space, args.device, args.out_format) == (
        "simple", "auto", "cpu", "auto")


def test_unknown_model_rejected():
    with pytest.raises(SystemExit):
        cli.main(["run", "-i", "a.mnc", "-o", "out", "--model", "extra"])


def test_tool_env_puts_prefix_first_and_keeps_caller_path():
    env = minctools.tool_env({"PATH": "/usr/bin:/bin", "MINC_COMPRESS": "0"})
    path = env["PATH"].split(os.pathsep)
    assert path[:2] == [os.path.join(sys.prefix, "pipeline"), os.path.join(sys.prefix, "bin")]
    assert path[2:] == ["/usr/bin", "/bin"]
    assert env["MINC_TOOLKIT"] == sys.prefix
    assert env["PERL5LIB"].startswith(os.path.join(sys.prefix, "perl"))
    assert env["MINC_COMPRESS"] == "0"  # caller's choice wins
    assert env["MINC_FORCE_V2"] == "1"


def test_tool_env_is_idempotent():
    once = minctools.tool_env({"PATH": "/usr/bin"})
    assert minctools.tool_env(once) == once


def test_field_rejects_other_values():
    with pytest.raises(SystemExit):
        cli.main(["run", "-i", "a.mnc", "-o", "out", "--field", "7"])


def test_unwritable_output_is_usage_error_not_traceback(tmp_path, caplog):
    scan = tmp_path / "s.mnc"
    scan.write_bytes(b"")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        code = cli.main(["run", "-i", str(scan), "-o", str(locked / "out" / "deeper"), "--dry-run"])
    finally:
        locked.chmod(0o700)
    assert code == cli.EXIT_USAGE
    assert "no permission" in caplog.text


def test_output_that_is_a_file_is_rejected(tmp_path, caplog):
    scan = tmp_path / "s.mnc"
    scan.write_bytes(b"")
    code = cli.main(["run", "-i", str(scan), "-o", str(scan), "--dry-run"])
    assert code == cli.EXIT_USAGE and "not a directory" in caplog.text
