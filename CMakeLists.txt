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

# I/O: FASTA, .maple, identical sequences, Newick (plain C++, no CUDA)
add_library(p2vbme_io STATIC src/fasta.cpp src/maple.cpp src/dedup.cpp src/newick.cpp src/io.cpp)
target_include_directories(p2vbme_io PUBLIC src)
target_compile_options(p2vbme_io PRIVATE -O3)
target_link_libraries(p2vbme_io PUBLIC p2vbme_core)

# Sequence simulation along a tree (tests, pipeline tools)
add_library(p2vbme_simulations STATIC src/simulations.cpp)
target_include_directories(p2vbme_simulations PUBLIC src)
target_compile_options(p2vbme_simulations PRIVATE -O3 -Wall -Wextra)
target_link_libraries(p2vbme_simulations PUBLIC p2vbme_core)

# Distance models: names, pair counts; the per-pair formulas are inline in the headers (plain C++, no CUDA)
add_library(p2vbme_distances STATIC src/substitution_models.cpp src/eigen_dist.cpp)
target_include_directories(p2vbme_distances PUBLIC src)
target_compile_options(p2vbme_distances PRIVATE -O3 -Wall -Wextra)

# JSON writer for the run report (plain C++, no CUDA)
add_library(p2vbme_json STATIC src/json_writer.cpp)
target_include_directories(p2vbme_json PUBLIC src)
target_compile_options(p2vbme_json PRIVATE -O3 -Wall -Wextra)

find_package(cxxopts CONFIG REQUIRED)

# The method: infer(), start tree, search, branch lengths (host), the CPU device and, with CUDA, the GPU device
# (all CUDA code is in device_gpu.cu)
add_library(p2vbme STATIC src/inference.cpp src/start_tree.cpp src/search.cpp src/branch_lengths.cpp
            src/report.cpp src/version.cpp src/device_cpu.cpp)
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

# Command-line tool (GPU and CPU, or CPU only without CUDA)
add_executable(p2v-bme src/cli.cpp)
target_compile_options(p2v-bme PRIVATE -O3)
target_link_libraries(p2v-bme PRIVATE p2vbme p2vbme_io cxxopts::cxxopts OpenMP::OpenMP_CXX)
install(TARGETS p2v-bme)

# ---- reference library (C99), as a shared library for the Python helpers --------------------------------
add_subdirectory(src/reference)

# ---- tests ----------------------------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
