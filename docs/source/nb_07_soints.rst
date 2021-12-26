Reading integrals in a spin orbital basis
=========================================

In this tutorial, you will learn how to read access integrals in a spin
orbital basis from python. These integrals can be used in pilot
implementations of quantum chemistry methods. By the end of this
tutorial you will know how to read the integrals and compute the
Hartree-Fock energy. For an implementation of MP2 based on the spin
orbital integrals see the file
``tests/pytest/helpers/test_spinorbital.py`` in the forte directory.

Forte assumes that the spin orbital basis :math:`\{ \psi_{p} \}` is
organized as follows

.. math::


   \underbrace{\phi_{0,\alpha}}_{\psi_0},
   \underbrace{\phi_{0,\beta}}_{\psi_1},
   \underbrace{\phi_{1,\alpha}}_{\psi_2},
   \underbrace{\phi_{1,\beta}}_{\psi_3},
   \ldots

To read the one-electron integrals
:math:`h_{pq} = \langle \psi_p | \hat{h} | \psi_q \rangle` we use the
function ``spinorbital_oei``. This function takes as arguments a
``ForteIntegrals`` object and two lists of integers, ``p`` and ``q``,
that specify the indices of the bra and ket **spatial orbitals**. For
example, if we want the integrals over the bra functions
:math:`\psi_0,\psi_1,\psi_3` and ket functions :math:`\psi_5,\psi_6` we
can write the following code

.. code:: python

       p = [0,1,3]
       q = [5,6]
       h = forte.spinorbital_oei(ints, p, q)

To read the two-electron antisymmetrized integrals in physicist notation
:math:`\langle pq \| rs \rangle` we use the function
``spinorbital_tei``, passing four list that corresponds to the range of
the indices ``p``, ``q``, ``r``, and ``s``.

.. code:: python

       p = [0,1]
       q = [0,1]
       r = [2,3]
       s = [2,3]    
       v = forte.spinorbital_tei(ints, p, q, r, s)

To compute the SCF energy we evaluate the expression

.. math::


   E = V_\mathrm{NN} + \sum_{i}^\mathrm{docc} h_{ii} + \frac{1}{2} \sum_{ij}^\mathrm{docc} \langle ij \| ij \rangle

where :math:`V_\mathrm{NN}` is the nuclear repulsion energy. To evaluate
this expression we only need the one- and two-electron integral blocks
that corresponds to the doubly occupied orbitals.

Preparing the orbitals via the ``utils.psi4_scf`` helper function
-----------------------------------------------------------------

To prepare an integral object it is necessary to first run a HF or
CASSCF computation.

Forte provides helper functions to run these computations using psi4. By
default **this function uses conventional integrals**.

.. code:: ipython3

    import math
    import numpy as np
    import forte
    import forte.utils
    
    geom = """
    O
    H 1 1.0
    H 1 1.0 2 104.5
    """
    
    escf_psi4, wfn = forte.utils.psi4_scf(geom=geom, basis='6-31G', reference='RHF')
    
    # grab the orbital occupation
    doccpi = wfn.doccpi().to_tuple()
    soccpi = wfn.soccpi().to_tuple()
    
    print(f'The SCF energy is {escf_psi4} [Eh]')
    print(f'SCF doubly occupied orbitals per irrep: {doccpi}')
    print(f'SCF singly occupied orbitals per irrep: {soccpi}')


.. parsed-literal::

    The SCF energy is -75.98015792193438 [Eh]
    SCF doubly occupied orbitals per irrep: (3, 0, 1, 1)
    SCF singly occupied orbitals per irrep: (0, 0, 0, 0)


Preparing the integral object
-----------------------------

To prepare the integrals, we use the helper function
``utils.prepare_forte_objects``. We pass the psi4 wave function object
(``wfn``) and specify the number of doubly occupied orbitals using the
SCF occupation from psi4. Virtual orbitals are automatically determined.

.. code:: ipython3

    mo_spaces={'RESTRICTED_DOCC' : doccpi, 'ACTIVE' : soccpi}
    forte_objects = forte.utils.prepare_forte_objects(wfn,mo_spaces)

The ``forte_objects`` returned is a dictionary, and we can access the
``ForteIntegral`` object using the key ``ints``. We store this object in
the variable ``ints``. We will also use the ``MOSpaceInfo`` object,
which is stored with the key ``mo_space_info``.

.. code:: ipython3

    ints = forte_objects['ints']
    mo_space_info = forte_objects['mo_space_info']

Preparing list of doubly occupied orbitals
------------------------------------------

From the ``MOSpaceInfo`` object we can find the list of doubly occupied
orbitals

.. code:: ipython3

    rdocc = mo_space_info.corr_absolute_mo('RESTRICTED_DOCC')
    print(f'List of doubly occupied orbitals: {rdocc}')


.. parsed-literal::

    List of doubly occupied orbitals: [0, 1, 2, 7, 9]


Preparing the core blocks of the Hamiltonian
--------------------------------------------

Here we call the functions that return the integrals in the spin orbital
basis. We store those in two variables, ``h`` and ``v``.

.. code:: ipython3

    h = forte.spinorbital_oei(ints, rdocc, rdocc)
    v = forte.spinorbital_tei(ints, rdocc, rdocc, rdocc, rdocc)
    
    with np.printoptions(precision=2, suppress=True):
        print(h)


.. parsed-literal::

    [[-32.98   0.    -0.58   0.    -0.19   0.     0.     0.     0.     0.  ]
     [  0.   -32.98   0.    -0.58   0.    -0.19   0.     0.     0.     0.  ]
     [ -0.58   0.    -7.78   0.    -0.3    0.     0.     0.     0.     0.  ]
     [  0.    -0.58   0.    -7.78   0.    -0.3    0.     0.     0.     0.  ]
     [ -0.19   0.    -0.3    0.    -6.8    0.     0.     0.     0.     0.  ]
     [  0.    -0.19   0.    -0.3    0.    -6.8    0.     0.     0.     0.  ]
     [  0.     0.     0.     0.     0.     0.    -7.07   0.     0.     0.  ]
     [  0.     0.     0.     0.     0.     0.     0.    -7.07   0.     0.  ]
     [  0.     0.     0.     0.     0.     0.     0.     0.    -6.5    0.  ]
     [  0.     0.     0.     0.     0.     0.     0.     0.     0.    -6.5 ]]


Evaluating the energy expression
--------------------------------

Here we add the three contributions to the energy and check the SCF
energy computed with psi4 and the one recomputed here

.. code:: ipython3

    escf = ints.nuclear_repulsion_energy()
    escf += np.einsum('ii->', h)
    escf += 0.5 * np.einsum('ijij->', v)
    
    print(f'The SCF energy is {escf_psi4} [Eh] (psi4)')
    print(f'The SCF energy is {escf} [Eh] (spin orbital integrals)')
    print(f'The difference is {escf_psi4 - escf} [Eh]')
    assert math.isclose(escf, escf_psi4)


.. parsed-literal::

    The SCF energy is -75.98015792193442 [Eh] (psi4)
    The SCF energy is -75.98015792193439 [Eh] (spin orbital integrals)
    The difference is -2.842170943040401e-14 [Eh]

