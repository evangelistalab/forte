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

1. Use Psi4 to generate a Makefile for Forte that is tailored to your current environment:
  ```
  cd src
  psi4 --new-plugin-makefile
  ```

2. After you have added this Makefile to the Forte directory you must add or
modify the following lines of your Makefile:
  ```
  NAME = forte
  
  # Define ambit directory
  AMBIT_DIR = <ambit install directory>
  
  # Need to link against Psi4 plugin library
  PSIPLUGIN = -L$(OBJDIR)/lib -lplugin -L$(AMBIT_DIR)/lib -lambit
  
  INCLUDES += -I$(AMBIT_DIR)/include
  
  PSITARGET = ../$(NAME).so
  ```

where (AMBIT_DIR) is the location of your Ambit install directory.
