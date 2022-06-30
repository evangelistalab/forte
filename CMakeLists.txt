cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Forte tests
option(ENABLE_ForteTests "Enable Forte tests." ON)

## 64bit implementation
if(MAX_DET_ORB)
    add_definitions(-DMAX_DET_ORB=${MAX_DET_ORB})
else()
    add_definitions(-DMAX_DET_ORB=64)
endif()

if (ENABLE_ForteTests)
  project (forte_tests)
  include_directories(${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/catch2/forte/catch2/single_include)
  add_executable(forte_tests
    tests/code/catch_amalgamated.cpp
    tests/code/test_determinant.cc
    tests/code/test_uint64.cc)

  project (forte_benchmarks)
  include_directories(${CMAKE_BINARY_DIR})
  add_executable(forte_benchmarks
    tests/benchmark/determinant_benchmark.cc)
endif (ENABLE_ForteTests)

# Add forte subdirectory
add_subdirectory (forte)
