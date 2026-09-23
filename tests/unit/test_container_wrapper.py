"""bin/hvr-cnn-container, checked through --print (no container tool needed)."""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "bin" / "hvr-cnn-container"


def printed(tmp_path, *args, engine="podman", image="hvr-cnn:test"):
    out = subprocess.run(
        ["bash", str(WRAPPER), "--engine", engine, "--image", image, "--print"] + list(args),
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        env=dict(os.environ, TMPDIR=str(tmp_path / "scratch")),
    )
    assert out.returncode == 0, out.stderr
    return shlex.split(out.stdout)


def volumes(cmd, flag):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


def test_podman_flags_and_mounts(tmp_path):
    (tmp_path / "scans").mkdir()
    cmd = printed(tmp_path, "run", "-i", "scans/a.mnc", "/elsewhere/b.nii.gz", "-o", "out")
    assert cmd[:3] == ["podman", "run", "--rm"]
    for flag in ("--read-only", "--userns=keep-id"):
        assert flag in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert cmd[cmd.index("--user") + 1] == "%d:%d" % (os.getuid(), os.getgid())
    v = volumes(cmd, "--volume")
    assert "%s:%s" % (tmp_path / "out", tmp_path / "out") in v           # output read-write
    assert "%s:%s:ro" % (tmp_path / "scans", tmp_path / "scans") in v   # inputs read-only
    assert "/elsewhere:/elsewhere:ro" in v
    assert (tmp_path / "out").is_dir()                                  # created before mounting
    # the hvr-cnn command line is passed through unchanged, after the image
    assert cmd[cmd.index("hvr-cnn:test") + 1:] == ["run", "-i", "scans/a.mnc", "/elsewhere/b.nii.gz",
                                                   "-o", "out"]
    assert cmd[cmd.index("--workdir") + 1] == str(tmp_path)


def test_each_directory_mounted_once_and_output_stays_writable(tmp_path):
    cmd = printed(tmp_path, "run", "-i", "a.mnc", "b.mnc", "-o", ".")
    v = volumes(cmd, "--volume")
    assert v.count("%s:%s" % (tmp_path, tmp_path)) == 1
    assert not any(x.startswith(str(tmp_path) + ":") and x.endswith(":ro") and x.count(":") == 2
                   and x.split(":")[0] == str(tmp_path) for x in v)


def test_csv_inputs_are_mounted(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "scans.csv").write_text("subject,input\ns1,rel/s1.mnc\ns2,/abs/dir/s2.mnc\n")
    cmd = printed(tmp_path, "run", "--csv", "t/scans.csv", "-o", "out")
    v = volumes(cmd, "--volume")
    assert "%s:%s:ro" % (tmp_path / "t" / "rel", tmp_path / "t" / "rel") in v   # relative to the table
    assert "/abs/dir:/abs/dir:ro" in v
    assert "%s:%s:ro" % (tmp_path / "t", tmp_path / "t") in v                   # the table itself


def test_tsv_with_bom(tmp_path):
    (tmp_path / "scans.tsv").write_bytes("\ufeffinput\tsubject\n/x/y/s.mnc\ts\n".encode("utf-8"))
    cmd = printed(tmp_path, "run", "--csv", "scans.tsv", "-o", "out")
    assert "/x/y:/x/y:ro" in volumes(cmd, "--volume")


def test_docker_has_no_keep_id_but_writable_tmp(tmp_path):
    cmd = printed(tmp_path, "run", "-i", "a.mnc", "-o", "out", engine="docker")
    assert "--userns=keep-id" not in cmd
    assert cmd[cmd.index("--tmpfs") + 1] == "/tmp"


@pytest.mark.parametrize("engine", ["apptainer", "singularity"])
def test_apptainer_binds_scratch_and_uses_docker_uri(tmp_path, engine):
    cmd = printed(tmp_path, "run", "-i", "a.mnc", "-o", "out", engine=engine, image="ghcr.io/x/hvr-cnn:1")
    assert cmd[:2] == [engine, "run"]
    for flag in ("--containall", "--cleanenv", "--no-home"):
        assert flag in cmd
    binds = volumes(cmd, "--bind")
    assert any(b.endswith(":/tmp") for b in binds) and any(b.endswith(":/var/tmp") for b in binds)
    assert "docker://ghcr.io/x/hvr-cnn:1" in cmd
    assert not list((tmp_path / "scratch").glob("hvr-cnn-container.*"))   # --print leaves nothing


def test_apptainer_sif_used_as_is(tmp_path):
    cmd = printed(tmp_path, "selftest", engine="apptainer", image="/images/hvr-cnn.sif")
    assert "/images/hvr-cnn.sif" in cmd and not any(a.startswith("docker://") for a in cmd)


def test_check_mounts_run_and_reference(tmp_path):
    cmd = printed(tmp_path, "check", "mine/run1", "--reference", "/ref/run1")
    v = volumes(cmd, "--volume")
    assert "%s:%s:ro" % (tmp_path / "mine" / "run1", tmp_path / "mine" / "run1") in v
    assert "/ref/run1:/ref/run1:ro" in v


def test_missing_csv_is_an_error(tmp_path):
    out = subprocess.run(["bash", str(WRAPPER), "--engine", "podman", "--print", "run", "--csv", "nope.csv",
                          "-o", "o"], cwd=str(tmp_path), stderr=subprocess.PIPE, universal_newlines=True)
    assert out.returncode == 2 and "no such file" in out.stderr
