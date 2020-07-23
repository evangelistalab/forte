![forte](lib/logos/forte_motologo_github.gif)

[![Build Status](https://travis-ci.org/evangelistalab/forte.svg?branch=master)](https://travis-ci.org/evangelistalab/forte)
[![Documentation Status](https://readthedocs.org/projects/forte/badge/?version=latest)](http://forte.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/evangelistalab/forte/branch/master/graph/badge.svg)](https://codecov.io/gh/evangelistalab/forte)

#### Code Authors: Evangelistalab

#### Manual (http://forte.readthedocs.io/)
#### Web: evangelistalab.org

Forte is an open-source plugin to Psi4 (https://github.com/psi4/psi4) that implements a variety of quantum chemistry methods
for strongly correlated electrons.

### Compilation

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4)

3. The tensor library Ambit (obtain it from https://github.com/jturney/ambit). Note that ambit is included in the conda distribution of psi4. So if you already have the latest version of psi4 installed there is no need to compile ambit.


Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte:

Forte may also be compiled by directly invoking CMake by following these instructions:

1. Run psi4 in the Forte folder
  ```
  psi4 --plugin-compile
  ```
 Psi4 will generate a CMake command for building Forte that looks like:
  ```
  cmake -C /usr/local/psi4/stage/usr/local/psi4/share/cmake/psi4/psi4PluginCache.cmake
        -DCMAKE_PREFIX_PATH=/usr/local/psi4/stage/usr/local/psi4 .
  ```
 
 2. Run the cmake command generated in 1. appending the location of Ambit's cmake files (via the `-Dambit_DIR option`):
 ```
  cmake -C /usr/local/psi4/stage/usr/local/psi4/share/cmake/psi4/psi4PluginCache.cmake
        -DCMAKE_PREFIX_PATH=/usr/local/psi4/stage/usr/local/psi4 .
        -Dambit_DIR=<ambit-bin-dir>/share/cmake/ambit
 ```
 
 3. Run make
 ```
  make
 ```

The following script automates steps 1 and 2 of the forte compilation process

```
#! /bin/tcsh

# Modify the following four parameters
set ambit_dir = /Users/fevange/Bin/ambit-Release/share/cmake/ambit/ # <- location of ambit
set srcdir = /Users/fevange/Source/forte   # <- location of forte source
set build_type = Release # <- Release, Release, or RelWithDebInfo

# Run cmake
cd $srcdir

set cmake_psi4 = `psi4 --plugin-compile`

$cmake_psi4 \
-Dambit_DIR=$ambit_dir \
-DCMAKE_BUILD_TYPE=$build_type \
-DMAX_DET_ORB=128 \
-DPYTHON_EXECUTABLE=/opt/anaconda3/bin/python \
-DENABLE_ForteTests=TRUE \
```
