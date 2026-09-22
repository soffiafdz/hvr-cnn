"""Thin wrapper over the external command-line tools.

The tools come from the same conda prefix as the running interpreter. They
need the variables that conda activation would set; the image has no
activated shell, so `tool_env` derives them from `sys.prefix` instead of
trusting the caller's environment.
"""

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

# Every external program the package may call; `selftest` checks them all.
REQUIRED_TOOLS = (
    # segmentation
    "itk_resample", "minccalc", "mincresample", "mincinfo",
    # NIfTI boundary
    "itk_convert",
    # native-space preprocessing
    "itk_minc_nonlocal_filter", "itk_morph", "mincdefrag", "autocrop",
    "minctracc", "N4BiasFieldCorrection", "volume_pol", "mincreshape",
    "xfminvert", "xfmconcat", "param2xfm",
    # QC picture
    "mincpik", "mincnorm", "minclookup", "magick", "montage",
)


def prefix():
    return sys.prefix


def tool_env(base=None):
    """Environment for child processes: the prefix's tools first on PATH."""
    p = prefix()
    env = dict(os.environ if base is None else base)
    path = [os.path.join(p, "pipeline"), os.path.join(p, "bin")]
    path += [d for d in env.get("PATH", "").split(os.pathsep) if d and d not in path]
    env["PATH"] = os.pathsep.join(path)
    perl = [os.path.join(p, "perl"), os.path.join(p, "pipeline")]
    perl += [d for d in env.get("PERL5LIB", "").split(os.pathsep) if d and d not in perl]
    env["PERL5LIB"] = os.pathsep.join(perl)
    env["MINC_TOOLKIT"] = p
    env["ANTSPATH"] = os.path.join(p, "bin")
    env["MNI_DATAPATH"] = os.path.join(p, "share")
    env.setdefault("MINC_FORCE_V2", "1")
    env.setdefault("MINC_COMPRESS", "4")
    env.setdefault("VOLUME_CACHE_THRESHOLD", "-1")
    return env


def which(tool):
    """Absolute path of `tool` as child processes will see it, or None."""
    return shutil.which(tool, path=tool_env()["PATH"])


def inside_prefix(path):
    real = os.path.realpath(path)
    return real.startswith(os.path.realpath(prefix()) + os.sep)


class ToolError(RuntimeError):
    """An external tool failed; the message carries its command and stderr."""


def run(cmd, **kwargs):
    """Run a tool; raise ToolError with the command and its stderr on failure."""
    cmd = [str(c) for c in cmd]
    log.debug("run: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, env=tool_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        **kwargs
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        raise ToolError("%s exited with status %d: %s | %s"
                        % (cmd[0], result.returncode, " ".join(cmd), " / ".join(tail) or "no output"))
    return result
