Overview
========

Forte is an open-source suite of state-of-the-art quantum chemistry
methods applicable to chemical systems with strongly correlated
electrons. The code is written as a plugin to Psi4 in C++ with C++17
functionality, and it takes advantage of shared memory parallelism
throughout.

Capabilities
------------

In general, Forte is composed of two types of methods: 1. Active space
solvers 1. Dynamical correlation solvers

==================================================== ============
Active space solvers                                 Abbreviation
==================================================== ============
Full/complete active space configuration interaction FCI/CASCI
Adaptive configuration interaction                   ACI
Projector configuration interaction                  PCI
==================================================== ============

==================================================== ============
Dynamical correlation solvers                        Abbreviation
==================================================== ============
Driven similarity renormalization group              DSRG
Second-order DSRG multireference perturbation theory DSRG-MRPT2
Third-order DSRG multireference perturbation theory  DSRG-MRPT3
Multireference DSRG with singles and doubles         MR-LDSRG(2)
==================================================== ============

Note that the active space solvers, notably FCI, ACI, and PCI can
operate within the full orbital basis defined by the user-selected basis
set. In this case, these methods also recover dynamical correlation.

Dependencies
------------

In order to run Forte, the following are required: - A Recent version of
Psi4 - CMake version 3.0 or higher - The tensor library Ambit

