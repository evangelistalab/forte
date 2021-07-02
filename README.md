![forte](lib/logos/forte_motologo_github.gif)

[![Build Status](https://travis-ci.org/evangelistalab/forte.svg?branch=master)](https://travis-ci.org/evangelistalab/forte)
[![Documentation Status](https://readthedocs.org/projects/forte/badge/?version=latest)](http://forte.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/evangelistalab/forte/branch/master/graph/badge.svg)](https://codecov.io/gh/evangelistalab/forte)

#### Code Authors: EvangelistaLab

#### Manual (http://forte.readthedocs.io/)
#### Web: evangelistalab.org

Forte is an open-source plugin to Psi4 (https://github.com/psi4/psi4) that implements a variety of quantum chemistry methods
for strongly correlated electrons.

## Download and compilation of Forte

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4)

3. The tensor library Ambit (obtain it from https://github.com/jturney/ambit). Note that ambit is included in the conda distribution of psi4. So if you already have the latest version of psi4 installed there is no need to compile ambit.


Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte.

### Download Forte

1. Open a terminal and change the current working directory to the location where you want to clone the Forte directory.
Let's assume this is the folder `src`.

2. Clone Forte from GitHub by pasting the following command:
```bash
git clone git@github.com:evangelistalab/forte.git
```
The repository will be cloned in the folder `src/forte`

### 1. Compilation via `setup.py` (recommended)

The most convenient way to compile forte is using the `setup.py` script. To compile Forte do the following:

1. From the `src` directory change to the forte directory `src/forte`
2. Tell `setup.py` where to find ambit, which can be done by creating the `src/forte/setup.cfg` file and adding the following lines
```tcsh
[CMakeBuild]
ambitpath=<ambit install dir>
```
or alternatively by setting the environmental variable `AMBITDIR` to point to the ambit install directory (note: there is no need to append `share/cmake/ambit`)
```tcsh
export AMBITPATH=<ambit install dir>
```
3. Compile forte by calling
```tcsh
python setup.py develop 
```
or for Debug mode
```tcsh
python setup.py build_ext --debug develop
```
This procedure will register forte within pip and you should be able to see forte listed just by calling
```tcsh
pip list
```
You can test that the path to Forte is set correctly by running python and importing forte:
```python
import forte
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
 ```tcsh
  make
 ```

#### Setting up the `PYTHONPATH`

If Forte is compiled with CMake, you will need to specify `PYTHONPATH` environment variable to make sure that it can be imported in python. Assuming that you cloned Forte from the folder `src` then you will have a folder named `src/forte`.
Your `PYTHONPATH` should then include `src/forte`
```bash
# in bash
export PYTHONPATH=<homedir>/src/forte:$PYTHONPATH 
```
This allows Forte to be imported correctly since the main `__init__.py` file for Forte is found at `src/forte/forte/__init__.py`


#### CMake script
The following script automates the Forte compilation process

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

make -j`getconf _NPROCESSORS_ONLN`
```

### Advanced compilation options

#### **Number of threads used to compile Forte**
To speed up compilation of Forte specify the number of threads to use for compilation.
This can be done in the `setup.cfg` file via
```tcsh
[CMakeBuild]
nprocs=<number of threads>
```
or when using CMake, compile Forte with the option -jn, for example, to compile with four threads
```tcsh
make -j4
```

#### Add configuration and build options
When using `setup.py` you can specify the `CMAKE_CONFIG_OPTIONS` and `CMAKE_BUILD_OPTIONS` passed internally to CMake in `setup.cfg`
```tcsh
[CMakeBuild]
cmake_config_options=...
cmake_build_options=...
```
These are convenient if you want to specify a different compiler from the one automatically detected by CMake.

#### **Maximum number of orbitals in the `Determinant` class**
By default, Forte is compiled assuming that the maximum number of orbitals that can be handled by codes that use the `Determinant` class is 64. To change this value modify the `setup.cfg` file to include
```tcsh
[CMakeBuild]
max_det_orb=<a multiple of 64>
```
or add the option
```tcsh
-DMAX_DET_ORB=<a multiple of 64>
```
if compiling with CMake.

#### **Enabling code coverage**
To enable compilation with code coverage activated, set the option `enable_codecov` to `ON` in the `setup.cfg` file
  ```tcsh
  [CMakeBuild]
  enable_codecov=ON
  ```
or add the option
  ```tcsh
  -DENABLE_CODECOV=ON
  ```
if compiling with CMake.
