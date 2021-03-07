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
        'scf_type': 'pk', # request conventional two-electron integrals
        'e_convergence': 12,
        'reference' : 'rohf'} # specify the reference
        )

After this, we set the input for the forte computation. We specify a multiplicity
equal to 3 and the :math:`B_1` irrep (``root_sym = 2``)::

    psi4.set_module_options(
    'FORTE', {
        'active_space_solver' : 'fci',
        'multiplicity': 3,            
        'root_sym': 2,
    }
    )
    psi4.energy('forte')

This input will run a FCI computation (since we have not specified the orbital spaces).
The output will return the energy and show the composition of the wave function::

    ==> Root No. 0 <==

    22a00000 b0 2000      0.69140575
    22b00000 a0 2000     -0.69140575

    Total Energy:     -38.998140978483, <S^2>: 2.000000

    ==> Energy Summary <==

    Multi.(2ms)  Irrep.  No.               Energy      <S^2>
    --------------------------------------------------------
       3  (  0)    B1     0      -38.998140978483   2.000000
    --------------------------------------------------------

Next we change the options to compute the lowest two :math:`1A_1` states.
We modify the multiplicity, the symmetry, and indicate that we want two
roots (``NROOTS = 2``)::

    psi4.set_module_options(
    'FORTE', {
        'multiplicity': 1,
        'root_sym': 0,
        'nroots' : 2,
    }
    )
    psi4.energy('forte')

The results of this computation is::

    ==> Root No. 0 <==

    22200000 00 2000      0.93105968
    22000000 20 2000     -0.28818136

    Total Energy:     -38.945534520808, <S^2>: 0.000000

    ==> Root No. 1 <==

    22000000 20 2000      0.91163553
    22200000 00 2000      0.26782667
    2ba00000 20 2000      0.13305423
    2ab00000 20 2000      0.13305423

    Total Energy:     -38.870842304841, <S^2>: 0.000000

    ==> Energy Summary <==

    Multi.(2ms)  Irrep.  No.               Energy      <S^2>
    --------------------------------------------------------
       1  (  0)    A1     0      -38.945534520808   0.000000
       1  (  0)    A1     1      -38.870842304841   0.000000
    --------------------------------------------------------
    
    
Computing a manifold of solutions of different symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next example shows how to perform a state-averaged CASSCF computation on two
electronic states of different symmetries. We stil conside methylene, and average
the lowest :math:`^3B_1` and :math:`^1A_1` states.
To begin, we use ROHF orbitals optimized for the lowest :math:`^3B_1`. However,
the final orbitals will optimize the average energy

.. math:: E_\mathrm{avg} = \frac{1}{2} \left(E_{^3B_1} + E_{^1A_1}\right)

In this example we use an active space that contains three :math:`^A_1` MOs,
and two MOs of :math:`^B_1` and :math:`^B_2` symmetry. We keep the lowest
:math:`^A_1` MO doubly occupied.
To specify the states, we use the ``AVG_STATE`` option::

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


