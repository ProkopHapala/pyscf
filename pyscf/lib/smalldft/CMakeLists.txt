# Copyright 2026 The PySCF Developers. All Rights Reserved.

add_library(smalldft SHARED small_grid.c)
add_dependencies(smalldft np_helper)

set_target_properties(smalldft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_include_directories(smalldft PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(smalldft np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
