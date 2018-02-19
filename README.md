![forte](lib/logos/forte_logo_github_2017.png)

[![Build Status](https://travis-ci.org/evangelistalab/forte.svg?branch=master)](https://travis-ci.org/evangelistalab/forte)
[![Documentation Status](https://readthedocs.org/projects/forte/badge/?version=latest)](http://forte.readthedocs.io/en/latest/?badge=latest)

#### Code Authors: Evangelistalab

#### Manual (http://forte.readthedocs.io/)
#### Web: evangelistalab.org

Forte is an open-source plugin to Psi4 (https://github.com/psi4/psi4) that implements a variety of quantum chemistry methods
for strongly correlated electrons.

Installation directions for Forte:

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. The tensor library Ambit (obtain it from https://github.com/jturney/ambit)

3. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4public)

Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte:

1. Run the setup script found in the Forte folder:
  ```
   python cmake_setup --psi4=<psi4 executable> --ambit-bindir=<Ambit binary installation dir>
  ```
  
  ```
optional arguments:
  -h, --help            show this help message and exit

PSI4 and CheMPS2 options:
  --psi4 PATH           The PSI4 executable. If this is left blank this script
                        will attempt to find PSI4 on your system. Failing that
                        it will not be able to compile FORTE. (default: None)
  --ambit-bindir PATH   The ambit binary installation directory. (default:
                        None)
  --chemps2-bindir PATH
                        The chemps2 binary installation directory. (default:
                        None)
  --mpi                 Whether to build the MPI part of code (default: False)
  --ga-bindir PATH      The GA install directory. (default: None)
   ```

2. Follow the instructions provided in the output of the `cmake_setup` script to compile Forte:
  ```
   configure step is done
   now you need to compile the sources:
   >>> cmake .
   >>> make
  ```

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
 
