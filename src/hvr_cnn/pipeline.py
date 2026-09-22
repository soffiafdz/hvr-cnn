"""`hvr-cnn run`: drive the scans through segmentation and write the output tree.

OUTDIR/
  volumes.tsv          one row per scan
  run.json             provenance and per-scan status
  <id>/<id>_space-stx_model-<model>_seg.mnc   labels on the input (stereotaxic) grid
"""

import csv
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, io, qc, segment, volumes
from .inputs import image_format
from .selftest import load_model

log = logging.getLogger(__name__)

ID_COLUMNS = ("id", "subject", "session", "group", "model", "input_space", "status")


class ScanError(Exception):
    """A scan could not be processed; the run continues with the next one."""


def _pick_device(requested):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ScanError("--device cuda requested but CUDA is not available in this build/host")
    return requested


def _count_labels(path):
    """Voxel counts per label of a MINC or NIfTI label file."""
    import numpy as np

    if str(path).endswith(".mnc"):
        data, _ = io.read_minc(path)
    else:
        import nibabel as nib

        data = np.asarray(nib.load(str(path)).dataobj)
    return volumes.count_labels(data)


def process_scan(scan, args, model, device, outdir, work_root):
    """Segment one scan. Returns (row for volumes.tsv, info for run.json)."""
    started = time.time()
    scan_dir = Path(outdir) / scan.id
    in_fmt = image_format(scan.path)
    out_fmt = in_fmt if args.out_format == "auto" else args.out_format
    if args.input_space not in ("auto", "stx"):
        raise ScanError("--input-space %s is not supported yet in this pre-release" % args.input_space)
    stem = "%s_space-stx_model-%s_seg" % (scan.id, args.model)
    seg_path = scan_dir / (stem + (".mnc" if out_fmt == "mnc" else ".nii.gz"))

    info = {"id": scan.id, "input": str(scan.path), "input_space": "stx"}
    if seg_path.exists() and not args.overwrite:
        info.update(status="skipped", reason="output exists (use --overwrite)")
        counts = _count_labels(seg_path)
    else:
        work = Path(work_root) / scan.id
        work.mkdir(parents=True, exist_ok=True)
        scan_dir.mkdir(parents=True, exist_ok=True)
        if in_fmt == "nii":
            t1 = work / (scan.id + "_input.mnc")
            try:
                io.nifti_to_minc(scan.path, t1)
            except io.ConversionError as exc:
                raise ScanError("cannot convert %s: %s" % (scan.path.name, exc)) from None
        else:
            t1 = scan.path
        if args.input_space == "auto":
            space, reason = segment.decide_space(t1)
            log.info("%s: input space decided as %s (%s)", scan.id, space, reason)
            if space != "stx":
                raise ScanError("looks like a native (unregistered) scan: %s. Native input is not supported "
                                "yet in this pre-release; if the scan is stereotaxic, pass --input-space stx" % reason)
        try:
            segment.check_stx_geometry(t1)
        except segment.GeometryError as exc:
            raise ScanError(str(exc)) from None
        p90 = segment.intensity_p90(t1)
        lo, hi = segment.INTENSITY_P90_RANGE
        if not lo <= p90 <= hi:
            info["warning"] = ("intensity p90 in the reference box is %.0f, expected %.0f-%.0f: the scan is "
                               "probably not intensity-normalised to the template" % (p90, lo, hi))
            log.warning("%s: %s", scan.id, info["warning"])
        labels_mnc = work / (stem + ".mnc")
        segment.segment_stx(t1, labels_mnc, args.model, model, work, device)
        counts = _count_labels(labels_mnc)
        tmp_out = work / seg_path.name
        if out_fmt == "nii":
            # on the input's own grid and axis order when the input was NIfTI, else RAS
            io.minc_labels_to_nifti(labels_mnc, tmp_out, like=scan.path if in_fmt == "nii" else None,
                                    description="hvr-cnn %s %s labels" % (__version__, args.model))
        else:
            tmp_out = labels_mnc
        if args.qc:  # before the move: for MINC output tmp_out is labels_mnc itself
            qc_path = scan_dir / ("%s_model-%s_qc.jpg" % (scan.id, args.model))
            title = "%s | %s | hvr-cnn %s" % (scan.id, args.model, __version__)
            qc.qc_picture(t1, labels_mnc, args.model, work / qc_path.name, work, title)
            shutil.move(str(work / qc_path.name), str(qc_path))
            info["qc"] = str(qc_path)
        shutil.move(str(tmp_out), str(seg_path))  # only complete outputs appear in OUTDIR
        info["status"] = "ok"
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    row = volumes.summarise(counts, args.model)
    info["seconds"] = round(time.time() - started, 1)
    info["outputs"] = [str(seg_path)]
    info["output_format"] = out_fmt
    if row["missing_labels"]:
        raise ScanError("segmentation incomplete: %d of %d expected labels absent (output kept for inspection: %s). "
                        "The input is probably not registered or not intensity-normalised to the template"
                        % (row["missing_labels"], len(volumes.expected_labels(args.model)), seg_path))
    return row, info


def write_volumes(path, rows):
    if not rows:
        return
    columns = list(ID_COLUMNS) + [c for c in rows[0] if c not in ID_COLUMNS]
    with open(str(path), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else ("%.6g" % value if key.endswith("HVR") else "%g" % value)
                else:
                    out[key] = "" if value is None else value
            writer.writerow(out)


def run(args, scans):
    """Process all scans; returns the number of failed scans."""
    import torch

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    threads = args.threads or _default_threads()
    torch.set_num_threads(threads)
    try:
        device = _pick_device(args.device)
    except ScanError as exc:
        log.error("%s", exc)
        return len(scans)
    model = load_model(args.model, device)
    log.info("model %s loaded on %s, %d thread(s)", args.model, device, threads)

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="hvr-cnn-")
        work_root = Path(tmp.name)

    started = datetime.now(timezone.utc)
    rows, infos, failed = [], [], 0
    try:
        for n, scan in enumerate(scans, 1):
            log.info("[%d/%d] %s", n, len(scans), scan.id)
            ident = {"id": scan.id, "subject": scan.subject, "session": scan.session,
                     "group": scan.group, "model": args.model, "input_space": "stx"}
            try:
                row, info = process_scan(scan, args, model, device, outdir, work_root)
                row = dict(ident, status=info["status"], **row)
                log.info("%s: %s | L HVR %s | R HVR %s | %ss", scan.id, info["status"],
                         _fmt(row["L_HVR"]), _fmt(row["R_HVR"]), info.get("seconds", "-"))
            except ScanError as exc:
                failed += 1
                info = {"id": scan.id, "input": str(scan.path), "status": "failed", "error": str(exc)}
                row = dict(ident, status="failed")
                log.error("%s: failed: %s", scan.id, exc)
            except Exception as exc:  # unexpected: record, keep going
                failed += 1
                info = {"id": scan.id, "input": str(scan.path), "status": "failed",
                        "error": "%s: %s" % (type(exc).__name__, exc), "traceback": traceback.format_exc()}
                row = dict(ident, status="failed")
                log.error("%s: failed: %s: %s", scan.id, type(exc).__name__, exc)
            rows.append(row)
            infos.append(info)
            if failed and args.fail_fast:
                log.error("stopping after the first failure (--fail-fast)")
                break
    finally:
        write_volumes(outdir / "volumes.tsv", rows)
        _write_run_json(outdir / "run.json", args, scans, infos, started, device, threads)
        if tmp is not None and not args.keep_work:
            tmp.cleanup()
        elif tmp is not None:
            log.info("intermediate files kept in %s", work_root)
    log.info("done: %d ok, %d failed, %d skipped -> %s", sum(i.get("status") == "ok" for i in infos), failed,
             sum(i.get("status") == "skipped" for i in infos), outdir)
    return failed


def _fmt(value):
    return "nan" if isinstance(value, float) and math.isnan(value) else "%.4f" % value


def _default_threads():
    from .cli import default_threads

    return default_threads()


def _write_run_json(path, args, scans, infos, started, device, threads):
    import torch

    record = {
        "hvr_cnn": __version__,
        "command": [os.path.basename(sys.argv[0])] + sys.argv[1:],
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "threads": threads,
        "image_digest": os.environ.get("HVR_CNN_IMAGE_DIGEST"),
        "scans": infos,
        "n_requested": len(scans),
    }
    with open(str(path), "w") as handle:
        json.dump(record, handle, indent=2)
