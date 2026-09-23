"""Command-line entry point: `hvr-cnn {run,selftest,check}`."""

import argparse
import logging
import os
import sys

from . import __version__
from .inputs import InputError, from_csv, from_paths, image_format

log = logging.getLogger("hvr_cnn")

EXIT_OK = 0
EXIT_SCAN_FAILED = 1  # the run finished but at least one scan failed
EXIT_USAGE = 2  # bad command line or bad inputs; nothing was processed
EXIT_UNAVAILABLE = 3  # selftest failed, or the requested feature is not available

# Options of the 0.0.x container, to tell its users what to type instead.
LEGACY_OPTIONS = {
    "-f": "--csv", "--csv_file": "--csv",
    "--input_img_path": "-i", "--output_dir": "-o", "--work_dir": "--work-dir",
    "-s": "a 'subject' column in --csv", "--subject_id": "a 'subject' column in --csv",
    "--session_id": "a 'session' column in --csv", "-g": "a 'group' column in --csv",
    "--group": "a 'group' column in --csv", "-m": "--model",
    "--qc_images": "not needed: QC pictures are written by default (--no-qc to skip)",
    "--clobber": "--overwrite",
    "--vols": None, "--volumes_csv": None, "--hc_vols": None, "--vc_vols": None,
    "--hvr": None, "--hvr_csv": None,
    "--seg_path": "-o", "--qc_path": "-o", "--hc_path": "-o", "--vc_path": "-o",
    "--hvr_path": "-o", "--has_header": None,
}

EXAMPLES = """\
examples:
  # one scan, already in stereotaxic space
  hvr-cnn run -i stx2_sub01_t1.mnc -o results

  # raw NIfTI scans, QC pictures, 8 threads
  hvr-cnn run -i sub-01_T1w.nii.gz sub-02_T1w.nii.gz -o results --input-space native --threads 8

  # a whole study from a table (header: input,subject,session,group), no QC pictures
  hvr-cnn run --csv scans.csv -o results --no-qc

  # check what would be done, without doing it
  hvr-cnn run --csv scans.csv -o results --dry-run

exit status: 0 success; 1 at least one scan failed (see OUTDIR/run.json);
             2 bad command line or inputs, nothing processed; 3 selftest failed.
"""


def default_threads():
    """CPUs this process may actually use: scheduler and cgroup limits, not the node size."""
    limits = [len(os.sched_getaffinity(0))] if hasattr(os, "sched_getaffinity") else [os.cpu_count() or 1]
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(var, "")
        if value.isdigit() and int(value) > 0:
            limits.append(int(value))
    return max(1, min(limits))


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="also log every external command")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="log warnings and errors only")

    parser = argparse.ArgumentParser(
        prog="hvr-cnn",
        description="Segment the hippocampus and the temporal horn of the lateral ventricle "
                    "on a T1w MRI and compute the hippocampal-to-ventricle ratio (HVR).",
        epilog="Run 'hvr-cnn <command> --help' for the options of a command.",
    )
    parser.add_argument("--version", action="version", version="hvr-cnn " + __version__)
    sub = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    run = sub.add_parser(
        "run", parents=[common], help="segment scans; write labels, volumes, HVR and QC",
        description="Segment one or more T1w scans. Nothing is written outside OUTDIR, "
                    "--work-dir and the system temporary directory.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = run.add_argument_group("input (one of -i / --csv is required)")
    source = inputs.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", nargs="+", metavar="T1",
                        help="T1w scans: .mnc, .mnc.gz, .nii or .nii.gz")
    source.add_argument("--csv", metavar="FILE",
                        help="CSV or TSV table with a header row: column 'input' (path, may be "
                             "relative to the table), optional 'subject', 'session', 'group'")
    inputs.add_argument("--input-space", default="auto",
                        choices=["auto", "stx", "native", "assemblynet"],
                        help="stx: already registered and intensity-normalised to ICBM152 2009c; "
                             "native: raw scan, preprocessed here; assemblynet: an AssemblyNet "
                             "'mni_t1' image; auto: decide per scan and log the decision "
                             "(default: %(default)s)")

    output = run.add_argument_group("output")
    output.add_argument("-o", "--output", required=True, metavar="OUTDIR",
                        help="output directory (created if missing)")
    output.add_argument("--out-format", default="auto", choices=["auto", "mnc", "nii"],
                        help="label format; auto = same as each input (default: %(default)s)")
    output.add_argument("--no-qc", dest="qc", action="store_false",
                        help="do not write the QC picture (written by default)")
    output.add_argument("--overwrite", action="store_true",
                        help="redo scans whose outputs exist (default: skip them, so an "
                             "interrupted run can be resumed with the same command)")

    proc = run.add_argument_group("processing")
    proc.add_argument("--model", default="simple", choices=["simple", "detailed"],
                      help="simple: hippocampus + temporal horn; detailed: head/body/tail of "
                           "both, plus amygdala (default: %(default)s)")
    proc.add_argument("--no-denoise", dest="denoise", action="store_false",
                      help="skip non-local-means denoising (native input; on by default, as in "
                           "the MNI longitudinal pipeline)")
    proc.add_argument("--denoise", dest="denoise", action="store_true", help=argparse.SUPPRESS)
    proc.add_argument("--field", type=float, default=3.0, choices=[1.5, 3.0], metavar="{1.5,3}",
                      help="scanner field strength, sets the N3 bias-field smoothness for native "
                           "input (default: %(default)g)")
    proc.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                      help="auto = cuda when available (default: %(default)s)")
    proc.add_argument("--threads", type=_positive_int, metavar="N",
                      help="CPU threads (default: what the scheduler/cgroup allows, here %d)"
                           % default_threads())

    runtime = run.add_argument_group("run control")
    runtime.add_argument("--work-dir", metavar="DIR",
                         help="directory for intermediate files (default: a temporary one)")
    runtime.add_argument("--keep-work", action="store_true", help="do not delete intermediate files")
    runtime.add_argument("--fail-fast", action="store_true",
                         help="stop at the first failed scan (default: continue, exit status 1)")
    runtime.add_argument("--dry-run", action="store_true",
                         help="validate the inputs, print the plan, process nothing")

    sub.add_parser("selftest", parents=[common],
                   help="check that this installation can run (needs no data)",
                   description="Check imports, external tools, bundled weights (one forward pass "
                               "per model) and temporary space. Exit status 0 when all pass.")

    check = sub.add_parser("check", parents=[common], help="verify the outputs of a run",
                           description="Verify that OUTDIR holds complete, plausible results.")
    check.add_argument("outdir", metavar="OUTDIR")
    check.add_argument("--reference", metavar="DIR",
                       help="compare against the results of another run (Dice, volumes)")
    return parser


def legacy_hint(argv):
    """Message for a 0.0.x style command line, or None."""
    used = [a.split("=")[0] for a in argv if a.split("=")[0] in LEGACY_OPTIONS]
    if not used or (argv and argv[0] in ("run", "selftest", "check")):
        return None
    lines = ["this looks like a command line for hvr_cnn 0.0.x; the interface has changed."]
    for opt in dict.fromkeys(used):
        new = LEGACY_OPTIONS[opt]
        lines.append("  %-18s -> %s" % (opt, new or "not needed: volumes and HVR are always written to OUTDIR/volumes.tsv"))
    lines.append("example: hvr-cnn run --csv scans.csv -o results    (see 'hvr-cnn run --help')")
    return "\n".join(lines)


def _plan(args, scans):
    threads = args.threads or default_threads()
    log.info("%d scan(s) -> %s | model %s | input space %s | device %s | %d thread(s)",
             len(scans), args.output, args.model, args.input_space, args.device, threads)
    for scan in scans:
        fmt = image_format(scan.path) if args.out_format == "auto" else args.out_format
        log.info("  %s  <- %s  (labels as %s)", scan.id, scan.path, fmt)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    hint = legacy_hint(argv)
    if hint:
        parser.exit(EXIT_USAGE, "hvr-cnn: %s\n" % hint)
    if not argv:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: run, selftest or check")
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "selftest":
        from . import selftest

        return EXIT_UNAVAILABLE if selftest.run() else EXIT_OK

    if args.command == "run":
        try:
            scans = from_csv(args.csv) if args.csv else from_paths(args.input)
        except InputError as exc:
            log.error("%s", exc)
            return EXIT_USAGE
        _plan(args, scans)
        if args.dry_run:
            log.info("dry run: nothing was processed")
            return EXIT_OK
        from . import pipeline

        return EXIT_SCAN_FAILED if pipeline.run(args, scans) else EXIT_OK

    if args.command == "check":
        from . import check

        return EXIT_SCAN_FAILED if check.main(args.outdir, args.reference) else EXIT_OK

    log.error("'%s' is not available in this pre-release (%s)", args.command, __version__)
    return EXIT_UNAVAILABLE


if __name__ == "__main__":
    sys.exit(main())
