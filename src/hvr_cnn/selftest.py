"""`hvr-cnn selftest`: is this installation able to run at all?

Needs no input data. Checks imports, external tools, bundled assets, that
both models load and run one patch, and that temporary space is writable.
"""

import importlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, minctools

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"
MODELS = {
    # model -> (weights, reference grid, output channels incl. background)
    "simple": ("ensemble_hcvc.pth", "ref_hcvc.mnc", 3),
    "detailed": ("ensemble_hcvc-ag.pth", "ref_hcvc-ag.mnc", 8),
}
REFERENCE_SHAPE = (119, 96, 140)  # x, y, z
PATCH = 96
IMPORTS = ("numpy", "torch", "nibabel", "minc2_simple",
           "model.ensemble", "model.vae2", "model.basic2", "model.util")


def load_model(name, device="cpu"):
    import torch

    weights = ASSETS / MODELS[name][0]
    # full-object pickle shipped inside the package, hence weights_only=False
    model = torch.load(str(weights), map_location="cpu", weights_only=False)
    return model.eval().to(device)


def _check_imports():
    for name in IMPORTS:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        yield name, True, version


def _check_tools():
    for tool in minctools.REQUIRED_TOOLS:
        path = minctools.which(tool)
        if path is None:
            yield tool, False, "not found"
        elif not minctools.inside_prefix(path):
            yield tool, False, "resolves outside the environment: " + path
        else:
            yield tool, True, path
    ref = ASSETS / MODELS["simple"][1]
    out = minctools.run(["mincinfo", "-dimlength", "xspace", ref]).stdout.strip()
    yield "mincinfo runs", out == str(REFERENCE_SHAPE[0]), "xspace=" + out
    out = minctools.run(["magick", "-version"]).stdout.splitlines()[0]
    yield "magick runs", "ImageMagick" in out, out


def _check_assets():
    from minc2_simple import minc2_file

    for name, (weights, ref, _) in MODELS.items():
        for f in (weights, ref):
            yield f, (ASSETS / f).is_file(), ""
        grid = minc2_file(str(ASSETS / ref))
        grid.setup_standard_order()
        shape = tuple(d.length for d in grid.representation_dims())
        grid.close()
        yield ref + " grid", shape == REFERENCE_SHAPE, "x".join(map(str, shape))
    yield "labels.map", (ASSETS / "labels.map").is_file(), ""


def _check_models():
    import torch

    for name, (_, _, channels) in MODELS.items():
        model = load_model(name)
        start = time.time()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, PATCH, PATCH, PATCH))["seg"]
        ok = tuple(out.shape) == (1, channels, PATCH, PATCH, PATCH)
        yield "model " + name, ok, "%s in %.1fs" % (tuple(out.shape), time.time() - start)


def _check_tmp():
    with tempfile.TemporaryDirectory(prefix="hvr-cnn-") as d:
        probe = Path(d) / "probe"
        probe.write_bytes(b"ok")
        yield "temporary directory", probe.read_bytes() == b"ok", tempfile.gettempdir()


def run():
    """Run every check, log one line each, return the number of failures."""
    import torch

    log.info("hvr-cnn %s | python %s | prefix %s", __version__,
             sys.version.split()[0], sys.prefix)
    log.info("torch %s | cuda build %s | cuda available %s | threads %d | uid %d | cwd %s",
             torch.__version__, torch.version.cuda, torch.cuda.is_available(),
             torch.get_num_threads(), os.getuid(), os.getcwd())
    failures = 0
    for group in (_check_imports, _check_tools, _check_assets, _check_models, _check_tmp):
        try:
            for what, ok, detail in group():
                log.log(logging.INFO if ok else logging.ERROR,
                        "%-4s %s %s", "ok" if ok else "FAIL", what, detail)
                failures += not ok
        except Exception as exc:  # a broken group must not hide the others
            log.error("FAIL %s: %s: %s", group.__name__, type(exc).__name__, exc)
            failures += 1
    log.log(logging.INFO if not failures else logging.ERROR,
            "selftest: %s", "passed" if not failures else "%d failure(s)" % failures)
    return failures
