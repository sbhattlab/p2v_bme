# Compiled tools of the paper's scripts (P2VBME_BUILD_PAPER_TOOLS), built into build/ next to p2v-bme

# JC69 sequences along a tree (simulate_data.py)
add_executable(sim_jc69_fast tools/sim_jc69_fast.c)
target_compile_options(sim_jc69_fast PRIVATE -O3)
target_link_libraries(sim_jc69_fast PRIVATE OpenMP::OpenMP_C m)

# RF distances of the pandemic-scale trees, all branches and resolved ones (run_benchmarks.py, run_baselines.py)
add_executable(eval_cmaple_rf tools/eval_cmaple_rf.cpp)
target_compile_options(eval_cmaple_rf PRIVATE -O3)

