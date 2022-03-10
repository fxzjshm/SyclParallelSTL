cmake_minimum_required (VERSION 3.2.2)
project (SyclSTL)

enable_testing()

option(PARALLEL_STL_BENCHMARKS "Build the internal benchmarks" OFF)

message(STATUS " Path to CMAKE source directory: ${CMAKE_SOURCE_DIR} ")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules/)
find_package(Threads REQUIRED)

set(HIPSYCL_TARGETS "omp")
find_package(hipSYCL REQUIRED)

set (CMAKE_CXX_STANDARD 20)

# PSTL specific
include_directories("include")

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (tests)

if (PARALLEL_STL_BENCHMARKS)
  add_subdirectory (benchmarks)
endif()
