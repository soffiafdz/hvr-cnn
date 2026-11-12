"""Turn `-i` paths or a `--csv` table into a validated list of scans.

Everything here runs before any processing, so that a typo in row 400 of a
CSV is reported immediately and not after six hours of segmentation.
"""

import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Optional

EXTENSIONS = (".mnc", ".mnc.gz", ".nii", ".nii.gz")
CSV_REQUIRED = "input"
CSV_OPTIONAL = ("subject", "session", "group", "xfm")


class InputError(ValueError):
    """A problem with what the user asked for; reported without a traceback."""


class Scan(NamedTuple):
    id: str  # unique within a run; names the output directory and files
    path: Path
    subject: Optional[str] = None
    session: Optional[str] = None
    group: Optional[str] = None
    xfm: Optional[Path] = None  # native -> stx transform of a stereotaxic input


def strip_extension(name):
    lower = name.lower()
    for ext in sorted(EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return name[: -len(ext)]
    raise InputError(
        "%s: unsupported file type (expected one of %s)" % (name, ", ".join(EXTENSIONS)))


def image_format(path):
    """'mnc' or 'nii', from the file name."""
    strip_extension(Path(path).name)
    return "mnc" if ".mnc" in Path(path).name.lower() else "nii"


def _safe(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")


def scan_id(path, subject=None, session=None):
    """`<subject>[_<session>]` when given, otherwise the file name without extension."""
    if subject:
        parts = [_safe(subject)] + ([_safe(session)] if session else [])
        ident = "_".join(parts)
    else:
        ident = _safe(strip_extension(Path(path).name))
    if not ident:
        raise InputError("%s: cannot derive an identifier" % path)
    return ident


def from_paths(paths):
    # absolute from here on, so logs, volumes.tsv and run.json name the exact file
    # (os.path.abspath normalises "." and ".." but does not follow symlinks)
    return _validated((Path(os.path.abspath(p)), None, None, None, None) for p in paths)


def from_csv(csv_path):
    """Read a CSV/TSV with a header row. Relative paths are relative to the table."""
    csv_path = Path(os.path.abspath(csv_path))  # so relative entries become absolute too
    if not csv_path.is_file():
        raise InputError("%s: no such file" % csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        if CSV_REQUIRED not in columns:
            raise InputError(
                "%s: needs a header row with a column named '%s' (optional: %s); found: %s"
                % (csv_path, CSV_REQUIRED, ", ".join(CSV_OPTIONAL), ", ".join(columns) or "nothing"))
        reader.fieldnames = columns
        rows = []
        for line, row in enumerate(reader, start=2):
            value = (row.get(CSV_REQUIRED) or "").strip()
            if not value:
                raise InputError("%s line %d: empty '%s'" % (csv_path, line, CSV_REQUIRED))
            extra = {k: (row.get(k) or "").strip() or None for k in CSV_OPTIONAL}
            xfm = extra["xfm"] and _from_table(csv_path, extra["xfm"])
            rows.append((_from_table(csv_path, value), extra["subject"], extra["session"], extra["group"], xfm))
    if not rows:
        raise InputError("%s: no rows" % csv_path)
    return _validated(rows)


def _from_table(csv_path, value):
    """A path from the table, absolute; relative entries are relative to the table."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(os.path.abspath(csv_path.parent / path))  # also resolves "../"
    return path


def _file_problem(path):
    if not path.is_file():
        return "%s: no such file" % path
    if not os.access(str(path), os.R_OK):
        return "%s: not readable" % path
    return None


def _validated(rows):
    """Build the scans from (path, subject, session, group, xfm) rows; report every problem at once."""
    scans, problems = [], []
    for path, subject, session, group, xfm in rows:
        try:
            image_format(path)
            scans.append(Scan(scan_id(path, subject, session), path, subject, session, group, xfm))
        except InputError as exc:
            problems.append(str(exc))
            continue
        problem = _file_problem(path)
        if problem:
            problems.append(problem)
        if xfm is not None:
            if not xfm.name.lower().endswith(".xfm"):
                problems.append("%s: the 'xfm' column needs a MINC linear transform (.xfm)" % xfm)
            else:
                problem = _file_problem(xfm)
                if problem:
                    problems.append(problem)
    for ident, n in Counter(s.id for s in scans).items():
        if n > 1:
            problems.append(
                "identifier '%s' occurs %d times; outputs would overwrite each other "
                "(use --csv with subject/session columns)" % (ident, n))
    if problems:
        shown = problems[:20]
        more = len(problems) - len(shown)
        raise InputError("\n  ".join(
            ["%d problem(s) with the inputs:" % len(problems)] + shown
            + (["... and %d more" % more] if more else [])))
    return scans
