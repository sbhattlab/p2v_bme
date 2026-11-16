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

# ---- tests ----------------------------------------------------------------------------------------------
include(CTest)
