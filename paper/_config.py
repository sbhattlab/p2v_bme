"""Paths and programs of the paper's scripts, relative to the repository root
(run the scripts from there)."""

from typing import Final

DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASETS_DIR: Final = f"{DEFAULT_DATA_DIR}/datasets"  # downloads and simulated sets
DEFAULT_RESULTS_DIR: Final = f"{DEFAULT_DATA_DIR}/results"  # raw results of the run_*.py scripts
# Markdown tables and figure data (make_tables.py)
DEFAULT_TABLES_DIR: Final = f"{DEFAULT_DATA_DIR}/tables"

DEFAULT_BUILD_DIR: Final = "build"  # cmake --preset default
VECME: Final = f"{DEFAULT_BUILD_DIR}/p2v-bme"
# paper/tools/bme_scores.cpp: exact BME lengths
BME_SCORES: Final = f"{DEFAULT_BUILD_DIR}/bme_scores"
# paper/tools/eval_cmaple_rf.cpp: pandemic-set RF
EVAL_CMAPLE_RF: Final = f"{DEFAULT_BUILD_DIR}/eval_cmaple_rf"
# paper/tools/sim_jc69_fast.c: JC69 simulator (simulate_data.py)
SIM_JC69: Final = f"{DEFAULT_BUILD_DIR}/sim_jc69_fast"
# paper/tools/reattach_left_out.cpp (run_silva.py)
REATTACH_LEFT_OUT: Final = f"{DEFAULT_BUILD_DIR}/reattach_left_out"
FASTME: Final = "fastme"  # comparison programs, from the paper environment
FASTTREE: Final = "FastTreeMP"  # FastTree with OpenMP
VERYFASTTREE: Final = "VeryFastTree"
