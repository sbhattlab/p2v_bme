# The reference library (C11, POSIX): the method on a stored N x N distance matrix (discrete gradient, Phylo2Vec
# encoding, the reference optimisers), the oracle for small-N tests. Not linked into p2v-bme.
# Its users compile bme_reference.c with their own flags; this shared build is for loading it from other programs.
set(BME_REFERENCE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bme_reference.c CACHE INTERNAL "")
set(BME_REFERENCE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

add_library(bme_reference SHARED bme_reference.c)
target_include_directories(bme_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# No -ffast-math: the numbers then match the tested build (test_bme_reference) bit for bit
target_compile_options(bme_reference PRIVATE -O3)
if(P2VBME_NATIVE)
  target_compile_options(bme_reference PRIVATE -march=native)
endif()
target_link_libraries(bme_reference PRIVATE m pthread)
