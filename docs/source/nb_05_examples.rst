Examples of advanced Forte computations
=======================================

Computing one or more FCI solutions of a given symmetry
-------------------------------------------------------

The first example shows how to perform separate computations on
different electronic states of methylene. We compute the lowest
:math:`^3B_1` state and the first two :math:`^1A_1` states. All of these
computations use ROHF orbitals optimized for the lowest :math:`^3B_1`.
Here we use the python API interface of Forte, but it is easy to
translate these examples to a psi4 psithon input.

This input (see ``examples/api/02_rohf_fci.py``) starts with the
geometry specification:

.. code:: python

   # examples/api/02_rohf_fci.py

   import psi4
   import forte

   mol = psi4.geometry("""
   0 3 # triplet state
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

   psi4.set_options(
       {
           'basis': 'DZ',
           'scf_type': 'pk', # <-- request conventional two-electron integrals
           'e_convergence': 12,
           'reference': 'rohf',
           'forte__active_space_solver': 'fci',
           'forte__restricted_docc': [1, 0, 0, 0],
           'forte__active': [3, 0, 2, 2],
           'forte__multiplicity': 3, # <-- triplet state
           'forte__root_sym': 2, # <-- B1 symmetry
       }
   )

Note that all options prefixed with ``forte__`` are specific to the
Forte computation. Here we specify a multiplicity equal to 3 and the
:math:`B_1` irrep (``root_sym = 2``). In this example we keep the lowest
:math:`A_1` MO doubly occupied (``'restricted_docc': [1, 0, 0, 0]``) and
use an active space that contains three :math:`A_1` MOs, and two MOs
each of :math:`B_1` and :math:`B_2` symmetry. Lastly, we run Forte:

.. code:: python

       psi4.energy('forte')

This input will run a CASCI computation (since we have not requested
orbital optimization). An example of how to request orbital optimization
can be found in the section *Computing a manifold of solutions of
different symmetry*. The output will return the energy and show the
composition of the wave function:

::

   ==> Root No. 0 <==

   2b0 a0 20      0.70326213
   2a0 b0 20     -0.70326213

   Total Energy:     -38.924726774489, <S^2>: 2.000000

   ==> Energy Summary <==

   Multi.(2ms)  Irrep.  No.               Energy      <S^2>
   --------------------------------------------------------
      3  (  0)    B1     0      -38.924726774489   2.000000
   --------------------------------------------------------

Next we change the options to compute the lowest two :math:`1A_1`
states. We modify the multiplicity, the symmetry, and indicate that we
want two roots (``NROOTS = 2``):

.. code:: python

       psi4.set_options({
           'forte__multiplicity': 1,
           'forte__root_sym': 0, # <-- A1 symmetry
           'forte__nroots' : 2}
       )
       psi4.energy('forte')

The results of this computation is:

::

   ==> Root No. 0 <==

   220 00 20      0.92134189
   200 20 20     -0.37537841

   Total Energy:     -38.866616413802, <S^2>: -0.000000

   ==> Root No. 1 <==

   200 20 20     -0.89364609
   220 00 20     -0.36032959
   ab0 20 20     -0.13675846
   ba0 20 20     -0.13675846

   Total Energy:     -38.800424868719, <S^2>: -0.000000

   ==> Energy Summary <==

   Multi.(2ms)  Irrep.  No.               Energy      <S^2>
   --------------------------------------------------------
      1  (  0)    A1     0      -38.866616413802  -0.000000
      1  (  0)    A1     1      -38.800424868719  -0.000000
   --------------------------------------------------------

State-averaged CASSCF with states of different symmetry
-------------------------------------------------------

The next example shows how to perform a state-averaged CASSCF
computation on two electronic states of different symmetries. We still
consider methylene, and average the lowest :math:`^3B_1` and
:math:`^1A_1` states. To begin, we use ROHF orbitals optimized for the
lowest :math:`^3B_1`. However, the final orbitals will optimize the
average energy :math:`E_\mathrm{avg} = \frac{1}{2} \left(E_{^3B_1} + E_{^1A_1}\right)`.
We use the same active space of the previous example,
but here to specify the state, we set the ``AVG_STATE`` option:

.. code:: python

   # examples/api/03_sa-casscf.py

   import psi4
   import forte

   psi4.geometry("""
   0 3
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

   psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf',
           'forte__job_type': 'mcscf_two_step',
           'forte__active_space_solver': 'fci',
           'forte__restricted_docc': [1, 0, 0, 0],
           'forte__active': [3, 0, 2, 2],
           'forte__avg_state': [[2, 3, 1], [0, 1, 1]]
           # [(B1, triplet, 1 state), (A1,singlet,1 state)]
       }
   )

   psi4.energy('forte')

The output of this computation (in ``examples/api/03_sa-casscf.out``)
shows the energy for both states in the following table:

::

       ==> Energy Summary <==

       Multi.(2ms)  Irrep.  No.               Energy      <S^2>
       --------------------------------------------------------
          1  (  0)    A1     0      -38.900217662950   0.000000
       --------------------------------------------------------
          3  (  0)    B1     0      -38.960623289646   2.000000
       --------------------------------------------------------

Using different mean-field guesses in CASSCF computations
---------------------------------------------------------

A common issue when running CASSCF computation problematic convergence
due to a poor orbital guess. By default, Forte’s CASSCF code uses a
Hartree-Fock guess on a state with the same charge and multiplicity of
the solution that we are seeking. The next example shows how to provide
initial orbitals from states with different multiplicity, different
charge and multiplicity, or obtained via DFT. Here we target the singlet
state of methylene, using the same active space of the previous example.

Guess with a different multiplicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first example, we will ROHF orbitals for the triplet state as a
starting guess for CASSCF. To specify a triplet state we modify the
geometry section. After the ROHF computation, we pass the option
``forte__multiplicity`` to instruct Forte to optimize a singlet state

.. code:: python

   # examples/api/04_casscf-triplet-guess.py

   import psi4
   import forte

   psi4.geometry("""
   0 3 # <-- here we specify a triplet state
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

   psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'rohf'})

   e, wfn = psi4.energy('scf',return_wfn=True)

   psi4.set_options({
           'forte__job_type': 'mcscf_two_step',
           'forte__multiplicity' : 1, # <-- to override multiplicity = 2 assumed from geometry
           'forte__active_space_solver': 'fci',
           'forte__restricted_docc': [1, 0, 0, 0],
           'forte__active': [3, 0, 2, 2],
       }
   )

   psi4.energy('forte',ref_wfn=wfn)

Guess with different charge and multiplicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the second example, we will ROHF orbitals for the doublet cation as a
starting guess for CASSCF. The relevant changes are made in the geometry
section, where we indicate a charge of +1 and multiplicity equal to 2:

.. code:: python

   # examples/api/05_casscf-doublet-guess.py

   # ...

   psi4.geometry("""
   1 2
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

and we also add options to fully specify the values of charge,
multiplicity, and :math:`M_S` used to perform the CASSCF computation

::

   psi4.set_options({
           'forte__job_type': 'mcscf_two_step',
           'forte__charge' : 0, # <-- to override charge = +1 assumed from geometry
           'forte__multiplicity' : 1, # <-- to override multiplicity = 2 assumed from geometry
           'forte__ms' : 0, # <-- to override ms = 1/2 assumed from geometry
           'forte__active_space_solver': 'fci',
           'forte__restricted_docc': [1, 0, 0, 0],
           'forte__active': [3, 0, 2, 2],
       }
   )

Guess based on DFT orbitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the last example, we pass DFT orbitals (triplet UB3LYP) as a starting
guess:

.. code:: python

   # examples/api/06_casscf-dft-guess.py

   # ...

   psi4.geometry("""
   0 3
   C
   H 1 1.085
   H 1 1.085 2 135.5
   """)

   psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference': 'uks'})

   e, wfn = psi4.energy('b3lyp',return_wfn=True)

   psi4.set_options({
           'forte__job_type': 'mcscf_two_step',
           'forte__charge' : 0, # <-- to override charge = +1 assumed from geometry
           'forte__multiplicity' : 1, # <-- to override multiplicity = 2 assumed from geometry
           'forte__ms' : 0, # <-- to override ms = 1/2 assumed from geometry
           'forte__active_space_solver': 'fci',
           'forte__restricted_docc': [1, 0, 0, 0],
           'forte__active': [3, 0, 2, 2],
       }
   )

The following is a numerical comparison of the convergence pattern of
these three computations and the default guess used by Forte (singlet
RHF, in this case)

::

   Singlet RHF guess (default guess)

                         Energy CI                    Energy Orbital
              ------------------------------  ------------------------------
       Iter.        Total Energy       Delta        Total Energy       Delta  Orb. Grad.  Micro
       ----------------------------------------------------------------------------------------
          1     -38.869758479716  0.0000e+00    -38.878577088058  0.0000e+00  6.0872e-07     9
          2     -38.894769773234 -2.5011e-02    -38.899566097104 -2.0989e-02  5.9692e-07     9
          3     -38.900644175245 -5.8744e-03    -38.900811142131 -1.2450e-03  2.8974e-07     7
          4     -38.900845440821 -2.0127e-04    -38.900853293556 -4.2151e-05  2.1094e-07     6
          5     -38.900856221599 -1.0781e-05    -38.900856382896 -3.0893e-06  3.5078e-07     5
          6     -38.900856468519 -2.4692e-07    -38.900856468929 -8.6033e-08  2.7648e-07     4
          7     -38.900856469070 -5.5024e-10    -38.900856469077 -1.4813e-10  3.4940e-08     3
       ----------------------------------------------------------------------------------------


   Triplet ROHF guess (04_casscf-triplet-guess.out)

                         Energy CI                    Energy Orbital
              ------------------------------  ------------------------------
       Iter.        Total Energy       Delta        Total Energy       Delta  Orb. Grad.  Micro
       ----------------------------------------------------------------------------------------
          1     -38.866616410911  0.0000e+00    -38.877770272313  0.0000e+00  1.8365e-06    10
          2     -38.894804745194 -2.8188e-02    -38.899492369417 -2.1722e-02  1.8438e-06     9
          3     -38.900608150627 -5.8034e-03    -38.900797672192 -1.3053e-03  1.1877e-07     8
          4     -38.900840657824 -2.3251e-04    -38.900851568289 -5.3896e-05  2.8346e-07     6
          5     -38.900856107378 -1.5450e-05    -38.900856342345 -4.7741e-06  3.5145e-07     5
          6     -38.900856468806 -3.6143e-07    -38.900856469025 -1.2668e-07  2.6265e-07     4
          7     -38.900856469077 -2.7063e-10    -38.900856469079 -5.3831e-11  3.0108e-08     3
       ----------------------------------------------------------------------------------------


   Doublet ROHF (examples/api/05_casscf-doublet-guess.out)

                         Energy CI                    Energy Orbital
              ------------------------------  ------------------------------
       Iter.        Total Energy       Delta        Total Energy       Delta  Orb. Grad.  Micro
       ----------------------------------------------------------------------------------------
          1     -38.819565876524  0.0000e+00    -38.862403924512  0.0000e+00  2.4525e-06    14
          2     -38.893973814690 -7.4408e-02    -38.899703281038 -3.7299e-02  2.7109e-06    11
          3     -38.900705647525 -6.7318e-03    -38.900828470211 -1.1252e-03  4.6263e-07     9
          4     -38.900850304213 -1.4466e-04    -38.900854820184 -2.6350e-05  2.4524e-07     8
          5     -38.900856305078 -6.0009e-06    -38.900856411175 -1.5910e-06  4.4290e-07     7
          6     -38.900856468122 -1.6304e-07    -38.900856468764 -5.7589e-08  1.6046e-07     7
          7     -38.900856469077 -9.5575e-10    -38.900856469079 -3.1550e-10  2.7174e-08     3
       ----------------------------------------------------------------------------------------    


   Unrestricted DFT (B3LYP) guess (examples/api/06_casscf-dft-guess.out)

                         Energy CI                    Energy Orbital
              ------------------------------  ------------------------------
       Iter.        Total Energy       Delta        Total Energy       Delta  Orb. Grad.  Micro
       ----------------------------------------------------------------------------------------
          1     -38.864953251693  0.0000e+00    -38.878418350280  0.0000e+00  1.6735e-06    11
          2     -38.893853205980 -2.8900e-02    -38.899336177723 -2.0918e-02  1.7239e-06     9
          3     -38.900627125514 -6.7739e-03    -38.900811355470 -1.4752e-03  1.6239e-07     8
          4     -38.900846596355 -2.1947e-04    -38.900853835219 -4.2480e-05  9.5895e-08     7
          5     -38.900856239152 -9.6428e-06    -38.900856388795 -2.5536e-06  1.0132e-07     6
          6     -38.900856468388 -2.2924e-07    -38.900856468891 -8.0096e-08  6.1277e-08     5
          7     -38.900856469072 -6.8447e-10    -38.900856469078 -1.8658e-10  2.4832e-08     3
       ----------------------------------------------------------------------------------------
