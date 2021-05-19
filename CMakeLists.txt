# CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# project(forte
#         VERSION 0.1
#         LANGUAGES CXX C)
# set(forte_AUTHORS      "Francesco A. Evangelista and Group")
# set(forte_DESCRIPTION  "Quantum Chemistry Methods for Strongly Correlated Electrons plugin to Psi4")
# set(forte_URL          "https://github.com/evangelistalab/forte")
# set(forte_LICENSE      "GPL-3.0+")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(TargetOpenMP_FIND_COMPONENTS "CXX")

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
add_subdirectory (forte)

#
# Forte tests
#
option(ENABLE_ForteTests "Enable Forte tests." ON)

# target_include_directories(forte PRIVATE .)

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
    tests/code/test_determinant.cc
    tests/code/test_uint64.cc)

  project (forte_benchmarks)
  include_directories(${CMAKE_BINARY_DIR})
  add_executable(forte_benchmarks
    tests/benchmark/determinant_benchmark.cc)
endif (ENABLE_ForteTests)

