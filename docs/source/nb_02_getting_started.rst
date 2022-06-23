Running Forte computations
==========================

In this section, we will look at the basics of running computations in
Forte.

There are two ways one can run Forte:

1. Using the plugin interface in Psi4.
2. Using Forte’s python API.

Running a FCI computation using the plugin interface
----------------------------------------------------

Let’s start by looking at how one can run Forte as a plugin to Psi4. The
following text input (see ``examples/plugin/01_fci/input.dat``) can be
used to run a FCI computation on the lithium dimer:

.. code:: python

   # examples/plugin/01_fci/input.dat

   import forte

   molecule {
   0 1
   Li 0.0 0.0 0.0
   Li 0.0 0.0 3.0
   units bohr
   }

   set {
     basis sto-3g
     scf_type pk
     e_convergence 10
   }

   set forte {
     active_space_solver fci
   }

   energy('forte')

To understand the structure of the input file, we will go over each
section of this input.

-  The file start with the ``import forte`` command, which loads the
   Forte module.

-  The next section specifies the molecular structure and the
   charge/multiplicity of the molecule. This section accepts inputs as
   specified in Psi4’s `Molecule and Geometry
   Specification <https://psicode.org/psi4manual/master/psithonmol.html>`__
   documentation, and accepts both Cartesian and Z-matrix coordinates

.. code:: python

   molecule {
   0 1
   Li 0.0 0.0 0.0
   Li 0.0 0.0 3.0
   units bohr
   }

-  The options block that follows passes options to Psi4. Here we set
   the basis (``basis sto-3g``), the type of SCF integral algorithm
   (``scf_type pk``, which uses conventional integrals), and the energy
   convergence threshold (``e_convergence 10``, equivalent to
   :math:`10^{-10}\; E_\mathrm{h}`)

.. code:: python

   set {
     basis sto-3g
     scf_type pk
     e_convergence 10
   }

-  The next section sets options specific to Forte. In a typical Forte
   job the user needs to specify two objects:

   -  An **active space solver**, used to treat static correlation
      effects. The active space solver finds a solution to the
      Schrödinger equation in the subset of active orbitals.
   -  A **dynamical correlation solver**, used to add dynamical electron
      correlation corrections on top of a wave function defined in the
      active space. To run a FCI computation, we only need to specify
      the active space solver, which is done by setting the option
      ``active_space_solver``:

.. code:: python

   set forte {
     active_space_solver fci
   }

-  The last line of the input calls the Psi4 energy method specifing
   that we want to run the ``forte`` module

.. code:: python

   energy('forte')

To run this computation we invoke psi4 on the command line

.. code:: bash

   >>>psi4 input.dat

This will run psi4 and produce the output file ``output.dat``, a copy of
which is available in the file ``examples/plugin/01_fci/output.dat``.
From this output, we can read the CI coefficient of the most important
determinants written in occupation number representation

::

       220 0 0 200 0 0      0.89740847 <-- coefficient
       200 0 0 200 0 2     -0.29206218
       200 0 0 200 2 0     -0.29206218
       200 0 0 220 0 0     -0.14391931

and a summary of the total energy of a state and the expectation value
of the spin squared operator (:math:`\hat{S}^2`)

::

       Multi.(2ms)  Irrep.  No.               Energy      <S^2>
       --------------------------------------------------------
          1  (  0)    Ag     0      -14.595808852754  -0.000000
       --------------------------------------------------------

Running a FCI computation using the python API
----------------------------------------------

The following input runs the same FCI computation discussed above using
the python API:

.. code:: python

   # examples/api/01_fci.py

   import psi4
   import forte

   psi4.geometry("""
   0 1
   Li 0.0 0.0 0.0
   Li 0.0 0.0 3.0
   units bohr
   """)

   psi4.set_options({
       'basis': 'sto-3g',                    # <-- set the basis set
       'scf_type': 'pk',                     # <-- request conventional two-electron integrals
       'e_convergence': 10,                  # <-- set the energy convergence
       'forte__active_space_solver' : 'fci'} # <-- specify the reference
       )

   psi4.energy('forte')

This python file mirrors the psi4 input file.

-  The file start with both the ``import psi4`` and ``import forte``
   commands, to load both the psi4 and Forte modules.

-  The next command creates a psi4 ``Molecule`` object calling the
   function ``psi4.geometry``. This object is stored in a default memory
   location and automatically used by psi4

.. code:: python

   psi4.geometry("""
   0 1
   Li 0.0 0.0 0.0
   Li 0.0 0.0 3.0
   units bohr
   """)

-  The options block that follows passes options to both Psi4 and Forte.
   Here we pass options as a python dictionary, prefixing options that
   are specific to Forte with ``forte__``:

.. code:: python

   psi4.set_options({
       'basis': 'sto-3g',                    # <-- set the basis set
       'scf_type': 'pk',                     # <-- request conventional two-electron integrals
       'e_convergence': 10,                  # <-- set the energy convergence
       'forte__active_space_solver' : 'fci'} # <-- specify the active space solver
       )

-  The last line of the python code calls the Psi4 energy method
   specifing that we want to run the ``forte`` module

.. code:: python

   psi4.energy('forte')

This computation is identical to the previous one and produces the exact
same output (see ``examples/plugin/01_fci.out``).

Passing options in Forte: psi4 interface vs. dictionaries (new)
---------------------------------------------------------------

In the previous sections, calcultation options were passed to Forte
through psi4. An alternative way to pass options is illustrated in the
following example (using the python API):

.. code:: python

   # examples/api/07_options_passing.py
   """Example of passing options as a dictionary in an energy call"""

   import psi4
   import forte

   psi4.geometry("""
   0 3
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

   psi4.set_options({
       'basis': 'DZ',
       'scf_type': 'pk',
       'e_convergence': 12,
       'reference': 'rohf',
   })

   forte_options = {
       'active_space_solver': 'fci',
       'restricted_docc': [1, 0, 0, 0],
       'active': [3, 0, 2, 2],
       'multiplicity': 3,
       'root_sym': 2,
   }

   efci1 = psi4.energy('forte', forte_options=forte_options)

   forte_options['multiplicity'] = 1
   forte_options['root_sym'] = 0
   forte_options['nroot'] = 2
   forte_options['root'] = 1

   efci2 = psi4.energy('forte', forte_options=forte_options)

-  Note how in this file we create a python dictionary
   (``forte_options``) and pass it to the ``energy`` function as the
   parameter ``forte_options``.

-  Passing options via a dictionary takes priority over passing options
   via psi4. This means that **any option previously passed via psi4 is
   ignored**.

-  This way of passing options is **safer** than the one based on psi4
   because, unless the user intentionally passes the same dictionary in
   the energy call, there is no memory effect where previously defined
   options have an effect on all subsequent calls to ``energy``.

-  Note how later in the file we call ``energy`` again but this time we
   modify the options directly by modifying the dictionary

.. code:: python

   forte_options['multiplicity'] = 1
   forte_options['root_sym'] = 0
   forte_options['nroot'] = 2
   forte_options['root'] = 1

Here we change the multiplicity and symmetry of the target state, and
compute two roots, reporting the energy of the second one.

This computation is identical to the previous one and produces the exact
same output (see ``examples/plugin/01_fci.out``).

Test cases and Jupyter Tutorials
--------------------------------

-  **Test cases**. Forte provides test cases for most of all methods
   implemented. This is a good place to start if you are new to Forte.
   Test cases based on Psi4’s plugin interface can be found in the
   ``<fortedir>/tests/methods`` folder. Test cases based on Forte’s
   python API can be found in the ``<fortedir>/tests/pytest`` folder.

-  **Jupyter Tutorials for Forte’s Python API**. Forte is designed as a
   C++ library with a lot of the classes and functionality exposed in
   Python via the ``pybind11`` library. Tutorials on how to use Forte’s
   API can be found
   `here <https://github.com/evangelistalab/forte/tree/master/tutorials%3E>`__.
