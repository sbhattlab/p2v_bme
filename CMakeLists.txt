cmake_minimum_required(VERSION 3.24)
project(vecme VERSION 0.1.0 LANGUAGES C CXX)

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
  set(VECME_CUDA_FOUND ON)
else()
  set(VECME_CUDA_FOUND OFF)
endif()
option(VECME_CUDA "Build the GPU code (needs the CUDA toolkit)" ${VECME_CUDA_FOUND})
# The prebuilt binary (the release preset): the CUDA runtime, libstdc++, libgcc and libgomp linked into vecme, so that
# it needs only glibc and, for the GPU, an NVIDIA driver.
option(VECME_STATIC_RUNTIME "Link the CUDA and C++ runtimes into vecme" OFF)
if(VECME_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

# The prebuilt binary links libgomp in too: its static archive in place of the shared library.
if(VECME_STATIC_RUNTIME)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a OUTPUT_VARIABLE VECME_LIBGOMP
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(OpenMP_gomp_LIBRARY ${VECME_LIBGOMP} CACHE FILEPATH "libgomp, linked into vecme" FORCE)
endif()
find_package(OpenMP REQUIRED COMPONENTS C CXX)

# The version is set once, in project(); development builds also record the git revision.
execute_process(
  COMMAND git describe --always --dirty
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE VECME_GIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

# ---- package ---------------------------------------------------------------------------------------
# Core types: the alignment and the tree, and the packing of a sequence (plain C++, no CUDA)
add_library(vecme_core STATIC src/core.cpp)
target_include_directories(vecme_core PUBLIC src)
target_compile_options(vecme_core PRIVATE -O3)

# I/O: FASTA, .maple, Newick (plain C++, no CUDA)
add_library(vecme_io STATIC src/fasta.cpp src/maple.cpp src/newick.cpp src/io.cpp)
target_include_directories(vecme_io PUBLIC src)
target_compile_options(vecme_io PRIVATE -O3)
target_link_libraries(vecme_io PUBLIC vecme_core)

# Sequence simulation along a tree (tests, pipeline tools)
add_library(vecme_simulations STATIC src/simulations.cpp)
target_include_directories(vecme_simulations PUBLIC src)
target_compile_options(vecme_simulations PRIVATE -O3 -Wall -Wextra)
target_link_libraries(vecme_simulations PUBLIC vecme_core)

# Distance models: names, pair counts; the per-pair formulas are inline in the headers (plain C++, no CUDA)
add_library(vecme_distances STATIC src/distances.cpp)
target_include_directories(vecme_distances PUBLIC src)
target_compile_options(vecme_distances PRIVATE -O3 -Wall -Wextra)

# GTR and GTR+GC fits: the exchangeabilities (and alpha) from sampled sequence pairs (plain C++, OpenMP, no CUDA).
# -ffp-contract=off: no fused multiply-adds (the default on ARM), so the fit gives the same bits on every CPU.
add_library(vecme_gtr_fit STATIC src/gtr_fit.cpp)
target_include_directories(vecme_gtr_fit PUBLIC src)
target_compile_options(vecme_gtr_fit PRIVATE -O3 -Wall -Wextra -ffp-contract=off)
target_link_libraries(vecme_gtr_fit PUBLIC vecme_core vecme_distances PRIVATE OpenMP::OpenMP_CXX)

# Before and after inference: base frequencies, the GTR or GTR+GC fit, identical sequences, the start tree (plain C++,
# no CUDA)
add_library(vecme_processing STATIC src/processing.cpp)
target_include_directories(vecme_processing PUBLIC src)
target_compile_options(vecme_processing PRIVATE -O3)
target_link_libraries(vecme_processing PUBLIC vecme_core vecme_distances vecme_gtr_fit)

# Bootstrap supports of a tree's branches: TBE and exact (plain C++, no CUDA)
add_library(vecme_support STATIC src/support.cpp)
target_include_directories(vecme_support PUBLIC src)
target_compile_options(vecme_support PRIVATE -O3 -Wall -Wextra)
target_link_libraries(vecme_support PUBLIC vecme_core)

# JSON writer for the run report (plain C++, no CUDA)
add_library(vecme_json STATIC src/json_writer.cpp)
target_include_directories(vecme_json PUBLIC src)
target_compile_options(vecme_json PRIVATE -O3 -Wall -Wextra)

find_package(cxxopts CONFIG REQUIRED)

# The method: infer(), start tree, search, branch lengths (host), the bootstrap, the CPU device and, with CUDA, the GPU
# device (all CUDA code is in device_gpu.cu)
add_library(vecme_lib STATIC src/inference.cpp src/start_tree.cpp src/search.cpp src/phylo2vec.cpp src/branch_lengths.cpp
            src/report.cpp src/version.cpp src/accelerator.cpp src/device.cpp src/device_cpu.cpp src/bootstrap.cpp)
set_source_files_properties(src/version.cpp PROPERTIES COMPILE_DEFINITIONS
                            "VECME_VERSION=\"${PROJECT_VERSION}\";VECME_GIT=\"${VECME_GIT}\"")
target_include_directories(vecme_lib PUBLIC src)
target_compile_options(vecme_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
                                      $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(vecme_lib PUBLIC vecme_core vecme_distances vecme_support OpenMP::OpenMP_CXX)
target_link_libraries(vecme_lib PRIVATE vecme_json)
if(VECME_CUDA)
  target_sources(vecme_lib PRIVATE src/device_gpu.cu)
  target_compile_definitions(vecme_lib PRIVATE VECME_WITH_CUDA)
  if(VECME_STATIC_RUNTIME)
    target_link_libraries(vecme_lib PUBLIC CUDA::cudart_static)
  else()
    target_link_libraries(vecme_lib PUBLIC CUDA::cudart)
  endif()
endif()

# The CPU code counts bits with __builtin_popcountll (the CPU device, the pair counts). On x86-64, -mpopcnt makes it the
# popcnt instruction (every x86-64 CPU since about 2008) instead of a libgcc call: CPU runs 1.3x (JC69) to 2.5x (TN93)
# faster on 10,000 x 1,000 sites, same trees.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(vecme_distances PRIVATE -mpopcnt)
  target_compile_options(vecme_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mpopcnt>)
endif()

# The command line (cxxopts): the arguments into Settings and the Report's command-line part
add_library(vecme_cli STATIC src/cli.cpp)
target_compile_options(vecme_cli PRIVATE -O3)
target_link_libraries(vecme_cli PUBLIC vecme_lib PRIVATE cxxopts::cxxopts)

# Command-line tool (GPU and CPU, or CPU only without CUDA)
add_executable(vecme src/main.cpp)
target_compile_options(vecme PRIVATE -O3)
target_link_libraries(vecme PRIVATE vecme_cli vecme_lib vecme_io vecme_processing OpenMP::OpenMP_CXX)
if(VECME_STATIC_RUNTIME)
  target_link_options(vecme PRIVATE -static-libstdc++ -static-libgcc)
endif()
install(TARGETS vecme)

# ---- the paper's tools ----------------------------------------------------------------------------------
option(VECME_BUILD_PAPER_TOOLS "Build the compiled tools of the paper's scripts (paper/tools)" OFF)
if(VECME_BUILD_PAPER_TOOLS)
  add_subdirectory(paper)
endif()

# ---- tests ----------------------------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
