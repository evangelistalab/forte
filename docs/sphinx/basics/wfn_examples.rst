.. _`sec:wfn_examples`:

Examples of Forte inputs
========================

.. sectionauthor:: Francesco A. Evangelista

Computing one or more solutions of a given symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first example shows how to perform separate computations on different electronic
states of methylene. We compute the lowest :math:`^3B_1` state and the first two :math:`^1A_1` states.
All of these computations use ROHF orbitals optimized for the lowest :math:`^3B_1`.
Here we use the python API interface of Forte, but it is easy to translate
these examples to a psi4 psithon input.

This input starts with the geometry specification::

    import psi4
    import forte

    mol = psi4.geometry("""
    0 3 # triplet state
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    psi4.set_options({
        'basis': 'DZ',
        'scf_type': 'pk', # <-- request conventional two-electron integrals
        'e_convergence': 12,
        'reference' : 'rohf'} # <-- specify the reference
        )

After this, we set the input for the forte computation. We specify a multiplicity
equal to 3 and the :math:`B_1` irrep (``root_sym = 2``).
In this example we keep the lowest :math:`^A_1` MO doubly occupied
(``'restricted_docc': [1, 0, 0, 0]``) and use an active space that contains three
:math:`^A_1` MOs, and two MOs each of :math:`^B_1` and :math:`^B_2` symmetry::

    psi4.set_module_options(
    'FORTE', {
        'active_space_solver' : 'fci',
        'restricted_docc': [1, 0, 0, 0],
        'active': [3, 0, 2, 2],           
        'multiplicity': 3,            
        'root_sym': 2, # <-- B1 symmetry
    }
    )
    psi4.energy('forte')

This input will run a CASCI computation (since we have not requested orbital optimization).
An example of how to request orbital optimization can be found in
:ref:`Computing a manifold of solutions of different symmetry`.
The output will return the energy and show the composition of the wave function::

    ==> Root No. 0 <==

    2b0 a0 20      0.70326213
    2a0 b0 20     -0.70326213

    Total Energy:     -38.924726774489, <S^2>: 2.000000

    ==> Energy Summary <==

    Multi.(2ms)  Irrep.  No.               Energy      <S^2>
    --------------------------------------------------------
       3  (  0)    B1     0      -38.924726774489   2.000000
    --------------------------------------------------------

Next we change the options to compute the lowest two :math:`1A_1` states.
We modify the multiplicity, the symmetry, and indicate that we want two
roots (``NROOTS = 2``)::

    psi4.set_module_options(
    'FORTE', {
        'multiplicity': 1,
        'root_sym': 0, # <-- A1 symmetry
        'nroots' : 2,
    }
    )
    psi4.energy('forte')

The results of this computation is::

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


    
Computing a manifold of solutions of different symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next example shows how to perform a state-averaged CASSCF computation on two
electronic states of different symmetries. We still consider methylene, and average
the lowest :math:`^3B_1` and :math:`^1A_1` states.
To begin, we use ROHF orbitals optimized for the lowest :math:`^3B_1`. However,
the final orbitals will optimize the average energy

.. math:: E_\mathrm{avg} = \frac{1}{2} \left(E_{^3B_1} + E_{^1A_1}\right)

We use the same active space of the previous example, but here to specify the state,
we set the ``AVG_STATE`` option::

    import psi4
    import forte

    mol = psi4.geometry("""
    0 3
    C
    H 1 1.085
    H 1 1.085 2 135.5
    """)

    psi4.set_options({'basis': 'DZ', 'scf_type': 'pk', 'e_convergence': 12, 'reference' : 'rohf'})
    psi4.set_module_options(
        'FORTE', {
            'job_type' : 'mcscf_two_step',
            'active_space_solver' : 'fci',
            'restricted_docc' : [1,0,0,0],
            'active' : [3,0,2,2],
            'avg_state' : [[2,3,1],[0,1,1]] # <-- [(B1, triplet, 1 state), (A1,singlet,1 state)]
        }
    )
    psi4.energy('forte')

The result of this computation is::

    ==> Energy Summary <==

    Multi.(2ms)  Irrep.  No.               Energy      <S^2>
    --------------------------------------------------------
       1  (  0)    A1     0      -38.900217662950   0.000000
    --------------------------------------------------------
       3  (  0)    B1     0      -38.960623289646   2.000000
    --------------------------------------------------------

