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

from . import __version__, io, minctools, preprocess, qc, segment, volumes
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
    ext = ".mnc" if out_fmt == "mnc" else ".nii.gz"
    stem = "%s_space-stx_model-%s_seg" % (scan.id, args.model)
    seg_path = scan_dir / (stem + ext)

    info = {"id": scan.id, "input": str(scan.path), "input_space": args.input_space}
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
        space = args.input_space
        asm_mask = preprocess.assemblynet_mask_for(scan.path)
        if space == "auto":
            if asm_mask is not None:
                space, reason = "assemblynet", "AssemblyNet mni_t1 with its mni_mask next to it"
            else:
                space, reason = segment.decide_space(t1)
            log.info("%s: input space decided as %s (%s)", scan.id, space, reason)
        info["input_space"] = space
        native_t1, xfm = None, None
        if space == "assemblynet":
            if asm_mask is None:
                raise ScanError("--input-space assemblynet needs an AssemblyNet mni_t1_*.nii.gz with its "
                                "mni_mask_* file in the same directory")
            mask_mnc = work / ("%s_mni_mask.mnc" % scan.id)
            if image_format(asm_mask) == "nii":
                io.nifti_to_minc(asm_mask, mask_mnc)
            else:
                mask_mnc = asm_mask
            normalised = work / ("%s_stx.mnc" % scan.id)
            try:
                _, expression = preprocess.assemblynet_to_stx(t1, mask_mnc, normalised, work)
            except preprocess.PreprocessError as exc:
                raise ScanError(str(exc)) from None
            info["intensity_expression"] = expression
            t1 = normalised
        if space == "native":
            native_t1 = t1
            xfm = work / ("%s_to-stx.xfm" % scan.id)
            t1 = work / ("%s_stx.mnc" % scan.id)
            started_pre = time.time()
            try:
                preprocess.native_to_stx(native_t1, t1, xfm, work, do_denoise=args.denoise)
            except preprocess.PreprocessError as exc:
                raise ScanError(str(exc)) from None
            info["preprocessing_seconds"] = round(time.time() - started_pre, 1)
            info["scale_factor"] = preprocess.xfm_scale_factor(xfm)
            log.info("%s: preprocessed in %.0fs, stx scale factor %.3f", scan.id, info["preprocessing_seconds"],
                     info["scale_factor"])
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
            # on the input's own grid and axis order when the input was a stereotaxic NIfTI, else RAS
            io.minc_labels_to_nifti(labels_mnc, tmp_out, like=scan.path if (in_fmt == "nii" and native_t1 is None) else None,
                                    description="hvr-cnn %s %s labels" % (__version__, args.model))
        else:
            tmp_out = labels_mnc
        if args.qc:  # before the move: for MINC output tmp_out is labels_mnc itself
            qc_path = scan_dir / ("%s_model-%s_qc.jpg" % (scan.id, args.model))
            title = "%s | %s | hvr-cnn %s" % (scan.id, args.model, __version__)
            qc.qc_picture(t1, labels_mnc, args.model, work / qc_path.name, work, title)
            shutil.move(str(work / qc_path.name), str(qc_path))
            info["qc"] = str(qc_path)
        if native_t1 is not None:
            native_labels = work / ("%s_space-native_model-%s_seg.mnc" % (scan.id, args.model))
            minctools.run(["itk_resample", labels_mnc, native_labels, "--clobber", "--labels", "--byte",
                           "--like", native_t1, "--transform", xfm, "--invert_transform"])
            native_out = scan_dir / (native_labels.stem + ext)
            if out_fmt == "nii":
                io.minc_labels_to_nifti(native_labels, work / native_out.name,
                                        like=scan.path if in_fmt == "nii" else None,
                                        description="hvr-cnn %s %s labels" % (__version__, args.model))
                shutil.move(str(work / native_out.name), str(native_out))
            else:
                shutil.move(str(native_labels), str(native_out))
            xfm_out = scan_dir / xfm.name
            shutil.copyfile(str(xfm), str(xfm_out))
            info["outputs"] = [str(seg_path), str(native_out), str(xfm_out)]
        shutil.move(str(tmp_out), str(seg_path))  # only complete outputs appear in OUTDIR
        info["status"] = "ok"
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    row = volumes.summarise(counts, args.model)
    if info.get("scale_factor"):
        for key in [k for k in row if k.endswith("_mm3")]:
            row[key + "_native"] = row[key] / info["scale_factor"]
    info["seconds"] = round(time.time() - started, 1)
    info.setdefault("outputs", [str(seg_path)])
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
