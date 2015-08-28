![forte](lib/forte_logo.png)
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

1. Use Psi4 to generate a Makefile for Forte that is tailored to you current environment
```
psi4 --new-plugin forte
```

2. After you have added this Makefile to the Forte directory you must add the following lines to your Makefile:
```
IPLUGIN = -L$(OBJDIR)/lib -lplugin -L(AMBIT_DIRECTORY)/obj/src -lambit
INCLUDES += -I(AMBIT_DIRECTORY)/include/ambit
```
where (AMBIT_DIRECTORY) is the location of your compiled version of Ambit.
