# Compiled tools of the paper's scripts (VECME_BUILD_PAPER_TOOLS), built into build/ next to vecme

# JC69 sequences along a tree (simulate_data.py)
add_executable(sim_jc69_fast tools/sim_jc69_fast.c)
target_compile_options(sim_jc69_fast PRIVATE -O3)
target_link_libraries(sim_jc69_fast PRIVATE OpenMP::OpenMP_C m)

# RF distances of the pandemic-scale trees, all branches and resolved ones (run_benchmarks.py, run_baselines.py)
add_executable(eval_cmaple_rf tools/eval_cmaple_rf.cpp)
target_compile_options(eval_cmaple_rf PRIVATE -O3)

# BME lengths of given trees, exact or as the paper's evaluator, and FastME's distance matrix (run_fastme.py,
# run_benchmarks.py)
add_executable(bme_scores tools/bme_scores.cpp)
target_compile_options(bme_scores PRIVATE -O3 -Wall -Wextra)
target_link_libraries(bme_scores PRIVATE vecme_io vecme_distances OpenMP::OpenMP_CXX)

# The sequences left out of the SILVA search put back next to their nearest searched sequence (run_silva.py)
add_executable(reattach_left_out tools/reattach_left_out.cpp)
target_compile_options(reattach_left_out PRIVATE -O3 -Wall -Wextra)
target_link_libraries(reattach_left_out PRIVATE vecme_io vecme_processing OpenMP::OpenMP_CXX)
