"""Plot constants: folders, output formats, the method colours shared by every figure,
and the figure styles."""

from typing import Final, TypedDict

DEFAULT_TABLES_DIR: Final = "data/tables"  # figure data written by paper/make_tables.py
DEFAULT_IMG_DIR: Final = "img"  # figure outputs (git-ignored)
FIGURE_FORMATS: Final = ("pdf", "svg", "png")  # every figure is saved once per format
STYLE: Final = "plots/paper.mplstyle"  # every figure

# Matplotlib metadata without timestamps, so that an unchanged figure is an unchanged file
SAVE_METADATA: Final = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

# The folder of each figure in img/: one per main figure, holding its supplementary figures
FIGURE_FOLDERS: Final = {
    "fig2": "fig2",
    "figS1": "fig2",  # S1-S3: every method by size
    "figS2": "fig2",
    "figS3": "fig2",
    "figS4": "fig2",  # bootstrap calibration
    "fig3": "fig3",
    "figS5": "fig3",  # SARS-CoV-2 shared branches
    "fig4": "fig4",
}


# One entry per method, used identically in every figure:
# key, legend label, colour, marker, filled, line style.
# Hue = program, shade = algorithm (light: ME, dark: ML), after ColorBrewer's Paired:
# FastTree blues, VeryFastTree greys, the MAPLE purples; DIPPER black; VecME vermilion.
# Checked for red-green colour blindness (Machado et al. 2009): methods drawn together stay apart.
# Markers pair up the same way (square and diamond, the two triangles), for colour-blind readers.
# METHODS: the methods of Figs 2c,d and S1-S3, in drawing order.
# PANDEMIC_METHODS: the other programs of Figs 3a and S5.
METHODS: Final = [
    ("FastTree-ME", "FastTree-ME", "#6BAED6", "s", True, "-"),
    ("FastTree-ML", "FastTree-ML", "#1F78B4", "D", True, "-"),
    ("VFT-ME", "VeryFastTree-ME", "#BDBDBD", "^", True, "-"),
    ("VFT-ML", "VeryFastTree-ML", "#636363", "v", True, "-"),
    ("DIPPER", "DIPPER", "#000000", "X", True, "-"),
    ("P2V", "VecME", "#D55E00", "o", True, "-"),
]
PANDEMIC_METHODS: Final = [
    ("MAPLE", "MAPLE", "#B294C7", "P", True, "--"),
    ("CMAPLE", "CMAPLE", "#6A3D9A", "h", True, "-."),
]


class Method(TypedDict):
    """How a method is drawn: legend label, colour, marker, filled marker, line style."""

    label: str
    color: str
    marker: str
    filled: bool
    ls: str


SPEC: Final = {
    m[0]: Method(label=m[1], color=m[2], marker=m[3], filled=m[4], ls=m[5])
    for m in [*METHODS, *PANDEMIC_METHODS]
}

# Fig. 2: every panel at one size, half the 180 mm page (a, b on the first row, c, d on the second)
FIG2_PANEL: Final = (88 / 25.4, 74 / 25.4)

# The datasets of the scaling figures (S1-S3) and their figure numbers
SCALING_FIGURES: Final = {"rnasim": "figS1", "jc10k": "figS2", "yule10k": "figS3"}

# VecME's datasets, for the figures that draw them as series (Fig. 3c takes their colours):
# dataset in the data, legend label, colour, marker, line style, filled marker, intensity (alpha).
# RNASim takes VecME's vermilion, the JC series teal, a hue no method uses
# (apart from vermilion for colour-blind readers and on tinted screens).
# The PASTA-tree series in full, the random-tree series lighter.
SCALING_SERIES: Final = [
    ("RNASim", "RNASim (PASTA)", SPEC["P2V"]["color"], "s", "--", True, 1.0),
    ("JC", "JC (PASTA)", "#1B9E77", "o", "-", True, 1.0),
    ("JC random tree", "JC random tree", "#1B9E77", "^", "-.", True, 0.55),
]

# Fig. 3: two rows of panels 74 mm high (a, b; c, d); within a row the main axes share their bottom
# and top, as fractions of the panel, so that panels placed side by side line up
FIG3_HEIGHT: Final = 74 / 25.4
FIG3_AXES_Y: Final = {"ab": (0.28, 0.96), "cd": (0.20, 0.84)}  # row -> (bottom, top)

# Fig. 3a: the pandemic programs, as named in maple.tsv, and their key in SPEC
PANDEMIC: Final = [("MAPLE", "MAPLE"), ("CMAPLE", "CMAPLE"), ("VecME", "P2V")]
