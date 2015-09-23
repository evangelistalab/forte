![forte](lib/logos/forte_logo_github.png)
#Forte
=============

#####Code Authors: Evangelistalab
#####Web: evangelistalab.org

Adaptive quantum chemistry methods

Installation directions for Forte:

Prior to the compilation of Forte you must first check to make sure you have the following:

1. CMake version 3.0 or higher

2. The tensor library Ambit (obtain it from https://github.com/jturney/ambit)

3. An updated version of Psi4 (obtain it from https://github.com/psi4/psi4public)

Once you have the current versions of Psi4, CMake, and Ambit, follow the following instructions to install Forte:

1. Run the setup script found in the forte folder:
  ```
   python setup --psi4=<psi4 executable> --ambit-bindir=<ambit installation dir>
  ```
2. Follow the instructions provided in the output of the `setup` script to compile forte:
  ```
   root directory: /Users/francesco/Source/forte
   psi4 executable: /Users/francesco/Source/psi4-obj-c++11-debug/bin/psi4
   ambit binary installation directory: /Users/francesco/Source/ambit-bin-release

   configure step is done
   now you need to compile the sources:
   >>> cd /Users/francesco/Source/forte/src
   >>> make
  ```
