"""QC picture: 7 x 4 montage of axial, sagittal (both sides) and coronal
slices through the medial temporal lobes, labels overlaid on the T1.

Same slice positions and colours as the original implementation; the title
uses an explicit font file because ImageMagick's default font in the
environment renders symbols.
"""

import sys
from pathlib import Path

from . import minctools
from .selftest import ASSETS

# medial-temporal box on the stereotaxic grid
BOX_START = (-65.0, -64.0, -55.0)
BOX_SIZE = (131, 98, 84)
# slice indices within the box, per mincpik orientation
SLICES = (
    ("transverse", [26, 30, 34, 38, 42, 44, 48]),
    ("sagittal", [28, 31, 34, 37, 40, 43, 47]),
    ("sagittal", [102, 99, 96, 93, 90, 87, 84]),
    ("coronal", [55, 50, 45, 40, 35, 30, 25]),
)
TILE = 200
# label value -> colour index in labels.map
QC_COLOURS = {
    "simple": {11: 4, 12: 6, 21: 8, 22: 13},
    "detailed": {111: 1, 112: 2, 113: 3, 121: 4, 122: 5, 123: 6, 130: 7,
                 211: 13, 212: 8, 213: 9, 221: 14, 222: 11, 223: 12, 230: 10},
}
FONT = Path(sys.prefix) / "fonts" / "DejaVuSans.ttf"


def qc_picture(t1, labels, model, output, work_dir, title):
    """Write `output` (JPEG) for stereotaxic `t1` and its `labels` (MINC)."""
    work = Path(work_dir)
    box = work / "qc_t1.mnc"
    minctools.run(["mincresample", t1, box, "-clobber", "-trilinear", "-q", "-fill", "-fillvalue", "0",
                   "-step", "1", "1", "1", "-start"] + [str(v) for v in BOX_START]
                  + ["-nelements"] + [str(v) for v in BOX_SIZE])
    lut = ";".join("%d %d" % kv for kv in QC_COLOURS[model].items())
    box_labels = work / "qc_labels.mnc"
    minctools.run(["itk_resample", labels, box_labels, "--clobber", "--labels", "--byte", "--like", box,
                   "--lut-string", lut])
    norm = work / "qc_norm.mnc"
    minctools.run(["mincnorm", "-clobber", "-cutoff", "0.5", box, norm])
    grey = work / "qc_grey.mnc"
    minctools.run(["minclookup", "-clobber", "-grey", "-range", "20", "90", norm, grey])
    colour = work / "qc_colour.mnc"
    minctools.run(["minclookup", "-clobber", "-discrete", "-lookup_table", ASSETS / "labels.map", box_labels, colour])
    merged = work / "qc_rgb.mnc"
    minctools.run(["mincmath", "-clobber", "-nocheck_dimensions", "-max", grey, colour, merged])
    tiles = []
    for n, (orientation, positions) in enumerate(SLICES):
        for pos in positions:
            tile = work / ("qc_%d_%s_%d.miff" % (n, orientation, pos))
            minctools.run(["mincpik", "-clobber", "-scale", "1", "-" + orientation, "-slice", str(pos), merged, tile])
            tiles.append(tile)
    montage = work / "qc_montage.miff"
    minctools.run(["montage", "-tile", "7x4", "-background", "grey10", "-geometry", "%dx%d+1+1" % (TILE, TILE)]
                  + tiles + [montage])
    minctools.run(["magick", montage, "-gravity", "NorthWest", "-background", "grey10", "-splice", "0x22",
                   "-font", FONT, "-pointsize", "16", "-fill", "white", "-annotate", "+4+3", title,
                   "-quality", "90", output])
    return output
