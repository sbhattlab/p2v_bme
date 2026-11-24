"""Plot constants: folders, output formats, the method colours shared by every figure,
and the figure styles."""

from typing import TypedDict

DEFAULT_TABLES_DIR = "data/tables"  # figure data written by paper/make_tables.py
DEFAULT_IMG_DIR = "img"  # figure outputs (git-ignored)
FIGURE_FORMATS = ("pdf", "svg")  # every figure is saved once per format
STYLE = "plots/paper.mplstyle"  # every figure

# Matplotlib metadata without timestamps, so that an unchanged figure is an unchanged file
SAVE_METADATA = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

# The folder of each figure in img/: one per main figure, holding its supplementary figures
FIGURE_FOLDERS = {
    "fig2": "fig2",
    "figS1": "fig2",  # S1-S3: every method by size
    "figS2": "fig2",
    "figS3": "fig2",
    "figS4": "fig2",  # alignment length
    "fig3": "fig3",
    "figS5": "fig3",  # SARS-CoV-2 variants and shared branches
}


# One entry per method, used identically in every figure:
# key, legend label, colour, marker, filled, line style.
# Hue encodes the family: vermilion = Phylo2VecME,
# purple = DIPPER, blues = ML searches, greys = ME searches.
METHODS = [
    ("FastTree-ME", "FastTree-ME", "#8A8A8A", "s", True, "-"),
    ("VFT-ME", "VeryFastTree-ME", "#4D4D4D", "^", True, "-"),
    ("FastTree-ML", "FastTree-ML", "#56B4E9", "v", True, "-"),
    ("VFT-ML", "VeryFastTree-ML", "#0072B2", "D", True, "-"),
    ("DIPPER", "DIPPER", "#8856A7", "X", True, "-"),
    ("P2V", "Phylo2VecME", "#D55E00", "o", True, "-"),
]


class Method(TypedDict):
    """How a method is drawn: legend label, colour, marker, filled marker, line style."""

    label: str
    color: str
    marker: str
    filled: bool
    ls: str


SPEC = {m[0]: Method(label=m[1], color=m[2], marker=m[3], filled=m[4], ls=m[5]) for m in METHODS}

# The datasets of the scaling figures (S1-S3) and their figure numbers
SCALING_FIGURES = {"rnasim": "figS1", "jc10k": "figS2", "yule10k": "figS3"}

# Fig. 2a,b: dataset in the data, legend label, colour, marker, line style, filled marker.
# The random-tree series is drawn last, dashed and with squares,
# so that it stays visible where it runs along the other JC series.
SCALING_SERIES = [
    ("JC", "JC (PASTA)", "#0072B2", "o", "-", True),
    ("RNASim", "RNASim (PASTA)", "#D55E00", "o", "-", True),
    ("JC random tree", "JC random tree", "#6A3D9A", "s", "--", True),
]

# Fig. 2c,d: the pandemic programs
PANDEMIC = [
    ("MAPLE", "#CC79A7", "o", "--"),
    ("CMAPLE", "#333333", "s", "-."),
    ("Phylo2VecME", "#009E73", "D", "-"),
]

# SARS-CoV-2 figures: the colours of the two programs
OURS_COLOUR, MAPLE_COLOUR = "#D55E00", "#CC79A7"
