![forte](lib/logos/forte_logo_github2.png)

#####Code Authors: Evangelistalab
#####Web: evangelistalab.org

Forte is an open-source plugin to Psi4 (https://github.com/psi4/psi4) that implements a variety of quantum chemistry methods
for strongly correlated electrons.

Installation directions for Forte:

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. The tensor library Ambit (obtain it from https://github.com/jturney/ambit)

3. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4public)

Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte:

1. Run the setup script found in the forte folder:
  ```
   python setup --psi4=<psi4 executable>
  ```
  
  ```
optional arguments:
  -h, --help            show this help message and exit

PSI4 and CheMPS2 options:
  --psi4 PATH           The PSI4 executable. If this is left blank this script
                        will attempt to find PSI4 on your system. Failing that
                        it will not be able to compile FORTE. (default: None)
  --chemps2-bindir PATH
                        The chemps2 binary installation directory. (default:
                        None)
  --mpi                 Whether to build the MPI part of code (default: False)
  --ga-bindir PATH      The GA install directory. (default: None)
   ```

2. Follow the instructions provided in the output of the `setup` script to compile forte:
  ```
   configure step is done
   now you need to compile the sources:
   >>> cd <path to forte>/src
   >>> make
  ```
