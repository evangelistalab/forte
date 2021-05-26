![forte](lib/logos/forte_motologo_github.gif)

[![Build Status](https://travis-ci.org/evangelistalab/forte.svg?branch=master)](https://travis-ci.org/evangelistalab/forte)
[![Documentation Status](https://readthedocs.org/projects/forte/badge/?version=latest)](http://forte.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/evangelistalab/forte/branch/master/graph/badge.svg)](https://codecov.io/gh/evangelistalab/forte)

#### Code Authors: Evangelistalab

#### Manual (http://forte.readthedocs.io/)
#### Web: evangelistalab.org

Forte is an open-source plugin to Psi4 (https://github.com/psi4/psi4) that implements a variety of quantum chemistry methods
for strongly correlated electrons.

## Compilation

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4)

3. The tensor library Ambit (obtain it from https://github.com/jturney/ambit). Note that ambit is included in the conda distribution of psi4. So if you already have the latest version of psi4 installed there is no need to compile ambit.


Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte.

### 1. Compilation via `setup.py` (recommended)

The most convenient way to compile forte is using the `setup.py` script. To compile Forte do the following:

1. Tell `setup.py` where to find ambit, which can be done either by setting the environmental variable `AMBITDIR` to point to the ambit install directory (note: there is no need to append `share/cmake/ambit`)

```tcsh
export AMBITPATH=<ambit install dir>
```
or by modifying the `<fortedir>/setup.cfg` file to include
```tcsh
[CMakeBuild]
ambitpath=<ambit install dir>
```

2. Compile forte by calling
```tcsh
fortedir> python setup.py develop 
```
or for Debug mode
```tcsh
fortedir> python setup.py build_ext --debug develop
```
This procedure will register forte within pip and you should be able to see forte listed just by calling
```tcsh
pip list
```

### 2. Compilation via CMake

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
-Dambit_DIR=$ambit_dir \ # remove this line if ambit is installed via conda
-DCMAKE_BUILD_TYPE=$build_type \
-DMAX_DET_ORB=128 \
-DPYTHON_EXECUTABLE=/opt/anaconda3/bin/python \
-DENABLE_ForteTests=TRUE \
```

### Advanced compilation options

- **Maximum number of orbitals in the `Determinant` class**.
By default, Forte is compiled assuming that the maximum number of orbitals that can be handled by codes that use the `Determinant` class is 64. To change this value modify the `<fortedir>/setup.cfg` file to include
```tcsh
[CMakeBuild]
max_det_orb=<a multiple of 64>
```
or add the option
```tcsh
-DMAX_DET_ORB=<a multiple of 64>
```
if compiling with CMake.

- **Enabling code coverage**. To enable compilation with code coverage activated, set the option `enable_codecov` to `ON` in the `<fortedir>/setup.cfg` file
```tcsh
[CMakeBuild]
enable_codecov=ON
```
or add the option
```tcsh
-DENABLE_CODECOV=ON
```
if compiling with CMake.
