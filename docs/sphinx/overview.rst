.. _`sec:overview`:

Overview 
========

.. sectionauthor:: Jeffrey B. Schriber 

Forte is an open-source suite of state-of-the-art quantum chemistry methods applicable
to chemical systems with strongly correlated electrons. The code is written as a plugin
to Psi4 in C++ with C++11 functionality, and it takes advantage of shared memory parallelism
throughout. 

Capabilities
------------

In general, Forte is composed of two types of methods:

#. Active space methods
    #. Full/complete active space configuration interaction (FCI)/(CASCI)
    #. Adaptive configuration interaction (ACI)
    #. Projector configuration interaction (PCI)
    #. Complete active space self-consistent field (CASSCF)
    #. Density Matrix Renormalization Group self-consistent field (DMRG-SCF)

#. Methods for dynamical correlation
    #. Driven similarity renormalization group (DSRG)
        #. DSRG-MRPT2
        #. DSRG-MRPT3
        #. MR-LDSRG(2) 

Note that the active space methods, notably FCI, ACI, and PCI, can operate within the full
orbital basis defined by the user-selected basis set. In this case, these methods also recover
dynamical correlation.

Dependencies
------------

In order to run Forte, the following are required:

#. A Recent version of Psi4
#. CMake version 3.0 or higher
#. The tensor library Ambit
