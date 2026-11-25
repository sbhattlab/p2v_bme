# Package tests, run by ctest. GPU tests carry the label `gpu` (ctest -LE gpu runs the others).
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)
# GPU tests also hold the resource lock `gpu`, so ctest -j runs them one at a time (they share one card).
set(GPU_TEST_PROPERTIES LABELS gpu RESOURCE_LOCK gpu)

add_executable(test_fasta test_fasta.cpp)
target_link_libraries(test_fasta PRIVATE vecme_io GTest::gtest_main)
gtest_discover_tests(test_fasta)

add_executable(test_maple test_maple.cpp)
target_link_libraries(test_maple PRIVATE vecme_io GTest::gtest_main)
gtest_discover_tests(test_maple)

add_executable(test_processing test_processing.cpp)
target_link_libraries(test_processing PRIVATE vecme_io vecme_processing vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_processing)

add_executable(test_newick test_newick.cpp)
target_link_libraries(test_newick PRIVATE vecme_io GTest::gtest_main)
gtest_discover_tests(test_newick)

add_executable(test_simulations test_simulations.cpp)
target_compile_options(test_simulations PRIVATE -Wall -Wextra)
target_link_libraries(test_simulations PRIVATE vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_simulations)

add_executable(test_distances test_distances.cpp)
target_compile_options(test_distances PRIVATE -Wall -Wextra)
target_link_libraries(test_distances PRIVATE vecme_distances vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_distances)

add_executable(test_eigen_dist test_eigen_dist.cpp)
target_compile_options(test_eigen_dist PRIVATE -Wall -Wextra)
target_link_libraries(test_eigen_dist PRIVATE vecme_distances vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_eigen_dist)

add_executable(test_support test_support.cpp)
target_compile_options(test_support PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_support PRIVATE vecme_support vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_support)

add_executable(test_bootstrap test_bootstrap.cpp)
target_compile_options(test_bootstrap PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_bootstrap PRIVATE vecme_lib vecme_processing vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_bootstrap DISCOVERY_TIMEOUT 60)

add_executable(test_gtr_fit test_gtr_fit.cpp)
target_compile_options(test_gtr_fit PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_gtr_fit PRIVATE vecme_gtr_fit vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_gtr_fit)

# infer() on the CPU (any build): brute force at small N and for SPR optimality, workers;
# InferenceGpu.* on the GPU: a search under every model, CPU against GPU
add_executable(test_inference test_inference.cpp)
target_compile_options(test_inference PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_inference PRIVATE vecme_lib vecme_processing vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_inference TEST_FILTER "-InferenceGpu.*" DISCOVERY_TIMEOUT 60)
if(VECME_CUDA)
  gtest_discover_tests(test_inference TEST_FILTER "InferenceGpu.*" PROPERTIES ${GPU_TEST_PROPERTIES}
                       DISCOVERY_TIMEOUT 60)
endif()

# The CPU device's window kernels (any build): move scores against exact BME changes, ties
add_executable(test_device_cpu test_device_cpu.cpp)
target_compile_options(test_device_cpu PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_device_cpu PRIVATE vecme_lib vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_device_cpu)

# The run's devices (any build): the settings select_devices refuses
add_executable(test_device test_device.cpp)
target_compile_options(test_device PRIVATE -Wall -Wextra)
target_link_libraries(test_device PRIVATE vecme_lib GTest::gtest_main)
gtest_discover_tests(test_device)

# Phylo2Vec encoding and decoding (src/phylo2vec): reference rows from the phylo2vec crate, round trips
add_executable(test_phylo2vec test_phylo2vec.cpp)
target_compile_options(test_phylo2vec PRIVATE -Wall -Wextra)
target_link_libraries(test_phylo2vec PRIVATE vecme_lib GTest::gtest_main)
gtest_discover_tests(test_phylo2vec)

# The search's building blocks (src/search.hpp) on random trees: summary, components, skeletons and their tables
add_executable(test_search test_search.cpp)
target_compile_options(test_search PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_search PRIVATE vecme_lib GTest::gtest_main)
gtest_discover_tests(test_search DISCOVERY_TIMEOUT 60)

# The start trees (src/start_tree.hpp): a failing device's error leaves the sample-and-place start
add_executable(test_start_tree test_start_tree.cpp)
target_compile_options(test_start_tree PRIVATE -Wall -Wextra)
target_link_libraries(test_start_tree PRIVATE vecme_lib GTest::gtest_main)
gtest_discover_tests(test_start_tree)

# O(N) branch lengths against the exact O(N^2) ones: CPU device; BranchLengthsGpu.* on the GPU
add_executable(test_branch_lengths test_branch_lengths.cpp)
target_compile_options(test_branch_lengths PRIVATE -O2 -Wall -Wextra)
target_link_libraries(test_branch_lengths PRIVATE vecme_lib vecme_processing vecme_simulations GTest::gtest_main)
gtest_discover_tests(test_branch_lengths TEST_FILTER "-BranchLengthsGpu.*" DISCOVERY_TIMEOUT 60)
if(VECME_CUDA)
  gtest_discover_tests(test_branch_lengths TEST_FILTER "BranchLengthsGpu.*" PROPERTIES ${GPU_TEST_PROPERTIES}
                       DISCOVERY_TIMEOUT 60)
endif()

add_executable(test_json_writer test_json_writer.cpp)
target_compile_options(test_json_writer PRIVATE -Wall -Wextra)
target_link_libraries(test_json_writer PRIVATE vecme_json GTest::gtest_main)
gtest_discover_tests(test_json_writer)

# --devices: GPU numbers and CPU threads
add_executable(test_cli test_cli.cpp)
target_compile_options(test_cli PRIVATE -Wall -Wextra)
target_link_libraries(test_cli PRIVATE vecme_cli GTest::gtest_main)
gtest_discover_tests(test_cli)

add_executable(test_random test_random.cpp)
target_include_directories(test_random PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(test_random PRIVATE -Wall -Wextra)
target_link_libraries(test_random PRIVATE GTest::gtest_main)
gtest_discover_tests(test_random)

add_executable(test_config test_config.cpp)
target_include_directories(test_config PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(test_config PRIVATE -Wall -Wextra)
target_link_libraries(test_config PRIVATE GTest::gtest_main)
gtest_discover_tests(test_config)

add_executable(test_parallel_errors test_parallel_errors.cpp)
target_include_directories(test_parallel_errors PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(test_parallel_errors PRIVATE -Wall -Wextra)
target_link_libraries(test_parallel_errors PRIVATE OpenMP::OpenMP_CXX GTest::gtest_main)
gtest_discover_tests(test_parallel_errors)

if(VECME_CUDA)
  # The GPU device: its distances under every model against the CPU formulas, through the edge-length pass; its errors
  add_executable(test_device_gpu test_device_gpu.cpp)
  target_compile_options(test_device_gpu PRIVATE -Wall -Wextra)
  target_link_libraries(test_device_gpu PRIVATE vecme_lib vecme_simulations GTest::gtest_main CUDA::cudart)
  gtest_discover_tests(test_device_gpu PROPERTIES ${GPU_TEST_PROPERTIES} DISCOVERY_TIMEOUT 60)
endif()

# The command line: tests/python/test_cli.py (pytest) runs vecme and checks its exit code, output and files.
# ctest runs it as cli (CPU tests) and, in CUDA builds, cli_gpu (GPU tests).
add_executable(make_test_fasta make_test_fasta.cpp)
target_link_libraries(make_test_fasta PRIVATE vecme_io vecme_simulations)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(CLI_PYTEST ${Python3_EXECUTABLE} -m pytest -p no:cacheprovider -v ${CMAKE_CURRENT_SOURCE_DIR}/python/test_cli.py)
set(CLI_PYTEST_ENV VECME_BIN=$<TARGET_FILE:vecme> VECME_MAKE_TEST_FASTA=$<TARGET_FILE:make_test_fasta>)
add_test(NAME cli COMMAND ${CLI_PYTEST} -m "not gpu")
set_tests_properties(cli PROPERTIES ENVIRONMENT "${CLI_PYTEST_ENV}")
if(VECME_CUDA)
  add_test(NAME cli_gpu COMMAND ${CLI_PYTEST} -m gpu)
  set_tests_properties(cli_gpu PROPERTIES ENVIRONMENT "${CLI_PYTEST_ENV}" ${GPU_TEST_PROPERTIES})
endif()
