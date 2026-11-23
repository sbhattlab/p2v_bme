cmake_minimum_required(VERSION 3.24)
project(p2vbme VERSION 0.1.0 LANGUAGES C CXX)

# Flags are set per target (-O3, OpenMP); no build type is set, so no -DNDEBUG or
# other flags are added. Everything is built into the top of the build folder (build/<program>), where the
# pipeline scripts look for it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# CUDA is used when a CUDA compiler is found. The GPU target defaults to the build
# machine's GPUs; override with -DCMAKE_CUDA_ARCHITECTURES=90 (H100), for example.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(P2VBME_CUDA_FOUND ON)
else()
  set(P2VBME_CUDA_FOUND OFF)
endif()
option(P2VBME_CUDA "Build the GPU code (needs the CUDA toolkit)" ${P2VBME_CUDA_FOUND})
if(P2VBME_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP REQUIRED COMPONENTS C CXX)

# The version is set once, in project(); development builds also record the git revision.
execute_process(
  COMMAND git describe --always --dirty
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE P2VBME_GIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

# ---- package ---------------------------------------------------------------------------------------
# Core types: the alignment and the tree, and the packing of a sequence (plain C++, no CUDA)
add_library(p2vbme_core STATIC src/core.cpp)
target_include_directories(p2vbme_core PUBLIC src)
target_compile_options(p2vbme_core PRIVATE -O3)

# I/O: FASTA, .maple, Newick (plain C++, no CUDA)
add_library(p2vbme_io STATIC src/fasta.cpp src/maple.cpp src/newick.cpp src/io.cpp)
target_include_directories(p2vbme_io PUBLIC src)
target_compile_options(p2vbme_io PRIVATE -O3)
target_link_libraries(p2vbme_io PUBLIC p2vbme_core)

# Sequence simulation along a tree (tests, pipeline tools)
add_library(p2vbme_simulations STATIC src/simulations.cpp)
target_include_directories(p2vbme_simulations PUBLIC src)
target_compile_options(p2vbme_simulations PRIVATE -O3 -Wall -Wextra)
target_link_libraries(p2vbme_simulations PUBLIC p2vbme_core)

# Distance models: names, pair counts; the per-pair formulas are inline in the headers (plain C++, no CUDA)
add_library(p2vbme_distances STATIC src/distances.cpp)
target_include_directories(p2vbme_distances PUBLIC src)
target_compile_options(p2vbme_distances PRIVATE -O3 -Wall -Wextra)

# GTR+GC fit: the exchangeabilities and alpha from sampled sequence pairs (plain C++, OpenMP, no CUDA).
# -ffp-contract=off: no fused multiply-adds (the default on ARM), so the fit gives the same bits on every CPU.
add_library(p2vbme_gtr_fit STATIC src/gtr_fit.cpp)
target_include_directories(p2vbme_gtr_fit PUBLIC src)
target_compile_options(p2vbme_gtr_fit PRIVATE -O3 -Wall -Wextra -ffp-contract=off)
target_link_libraries(p2vbme_gtr_fit PUBLIC p2vbme_core p2vbme_distances PRIVATE OpenMP::OpenMP_CXX)

# Before and after inference: base frequencies, the GTR+GC fit, identical sequences, the start tree (plain C++, no
# CUDA)
add_library(p2vbme_processing STATIC src/processing.cpp)
target_include_directories(p2vbme_processing PUBLIC src)
target_compile_options(p2vbme_processing PRIVATE -O3)
target_link_libraries(p2vbme_processing PUBLIC p2vbme_core p2vbme_distances p2vbme_gtr_fit)

# Bootstrap supports of a tree's branches: TBE and exact (plain C++, no CUDA)
add_library(p2vbme_support STATIC src/support.cpp)
target_include_directories(p2vbme_support PUBLIC src)
target_compile_options(p2vbme_support PRIVATE -O3 -Wall -Wextra)
target_link_libraries(p2vbme_support PUBLIC p2vbme_core)

# JSON writer for the run report (plain C++, no CUDA)
add_library(p2vbme_json STATIC src/json_writer.cpp)
target_include_directories(p2vbme_json PUBLIC src)
target_compile_options(p2vbme_json PRIVATE -O3 -Wall -Wextra)

find_package(cxxopts CONFIG REQUIRED)

# The method: infer(), start tree, search, branch lengths (host), the CPU device and, with CUDA, the GPU device
# (all CUDA code is in device_gpu.cu)
add_library(p2vbme STATIC src/inference.cpp src/start_tree.cpp src/search.cpp src/phylo2vec.cpp src/branch_lengths.cpp
            src/report.cpp src/version.cpp src/accelerator.cpp src/device.cpp src/device_cpu.cpp)
set_source_files_properties(src/version.cpp PROPERTIES COMPILE_DEFINITIONS
                            "P2VBME_VERSION=\"${PROJECT_VERSION}\";P2VBME_GIT=\"${P2VBME_GIT}\"")
target_include_directories(p2vbme PUBLIC src)
target_compile_options(p2vbme PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
                                      $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(p2vbme PUBLIC p2vbme_core p2vbme_distances OpenMP::OpenMP_CXX)
target_link_libraries(p2vbme PRIVATE p2vbme_json)
if(P2VBME_CUDA)
  target_sources(p2vbme PRIVATE src/device_gpu.cu)
  target_compile_definitions(p2vbme PRIVATE P2VBME_WITH_CUDA)
  target_link_libraries(p2vbme PUBLIC CUDA::cudart)
endif()

# The CPU code counts bits with __builtin_popcountll (the CPU device, the pair counts). On x86-64, -mpopcnt makes it the
# popcnt instruction (every x86-64 CPU since about 2008) instead of a libgcc call: CPU runs 1.3x (JC69) to 2.5x (TN93)
# faster on 10,000 x 1,000 sites, same trees.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(p2vbme_distances PRIVATE -mpopcnt)
  target_compile_options(p2vbme PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mpopcnt>)
endif()

# The command line (cxxopts): the arguments into Settings and the Report's command-line part
add_library(p2vbme_cli STATIC src/cli.cpp)
target_compile_options(p2vbme_cli PRIVATE -O3)
target_link_libraries(p2vbme_cli PUBLIC p2vbme PRIVATE cxxopts::cxxopts)

# Command-line tool (GPU and CPU, or CPU only without CUDA)
add_executable(p2v-bme src/main.cpp)
target_compile_options(p2v-bme PRIVATE -O3)
target_link_libraries(p2v-bme PRIVATE p2vbme_cli p2vbme p2vbme_io p2vbme_processing OpenMP::OpenMP_CXX)
install(TARGETS p2v-bme)

# ---- reference library (C99), as a shared library for the Python helpers --------------------------------
# Off by default, so that a build does not depend on the CPU of the machine that runs it (CI runners differ).
option(P2VBME_NATIVE "Build the reference library and its test for this machine's CPU (-march=native)" OFF)
add_subdirectory(src/reference)

# ---- tests ----------------------------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
