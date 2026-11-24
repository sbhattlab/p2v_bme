"""Paths and programs of the paper's scripts, relative to the repository root (run the scripts from there)."""

DEFAULT_DATA_DIR = "data"
DEFAULT_DATASETS_DIR = f"{DEFAULT_DATA_DIR}/datasets"  # downloads and simulated sets
DEFAULT_RESULTS_DIR = f"{DEFAULT_DATA_DIR}/results"  # raw results of the run_*.py scripts
DEFAULT_TABLES_DIR = f"{DEFAULT_DATA_DIR}/tables"  # LaTeX tables and figure data (make_tables.py)

DEFAULT_BUILD_DIR = "build"  # cmake --preset default
P2VBME = f"{DEFAULT_BUILD_DIR}/p2v-bme"
BME_SCORES = f"{DEFAULT_BUILD_DIR}/bme_scores"  # paper/tools/bme_scores.cpp: exact BME lengths
EVAL_CMAPLE_RF = (
    f"{DEFAULT_BUILD_DIR}/eval_cmaple_rf"  # paper/tools/eval_cmaple_rf.cpp: pandemic-set RF
)
SIM_JC69 = f"{DEFAULT_BUILD_DIR}/sim_jc69_fast"  # paper/tools/sim_jc69_fast.c: JC69 simulator (simulate_data.py)
REATTACH_LEFT_OUT = (
    f"{DEFAULT_BUILD_DIR}/reattach_left_out"  # paper/tools/reattach_left_out.cpp (run_silva.py)
)
FASTME = "fastme"  # comparison programs, from the paper environment
FASTTREE = "FastTreeMP"  # FastTree with OpenMP
VERYFASTTREE = "VeryFastTree"
