Forte Python API Tutorial (NEW)
===============================

Forte’s new python API allows the user to express a calculation as a
computational graph. Nodes on this graph do one of the following -
Provide inputs - Take inputs from other nodes and perform a
computational task

Creating the input node
-----------------------

The starting point for a Forte computation is an input object
(``Input``). The input can be created via a factory function
(``forte.solvers.solver_factory``)

.. code:: python

       from forte.solvers import solver_factory
       
       # define the molecular geometry (H2, r = 1 Å)
       zmat = """
       H
       H 1 1.0
       """
       
       # create the input node using the zmat geometry and the cc-pVDZ basis set
       input = solver_factory(molecule=zmat, basis='cc-pVDZ')

The object returned by ``solver_factory`` (``input``) is an input node
that contains a ``MolecularModel`` attribute responsible for generating
the Hamiltonian of this molecule/basis set combination. The ``input``
object can now be passed to a ``Solver`` node that will perform a
computation.

Hartree–Fock theory
-------------------

To run a Hartree–Fock (HF) computation on the molecule defined above the
user has to do the following:

1. Specify the electronic state
2. Create a Hartree–Fock solver object
3. Call the ``run()`` function

Here is an example that shows the full input for a HF computation:

.. code:: python

   from forte.solvers import solver_factory, HF

   xyz = """
   H 0.0 0.0 0.0
   H 0.0 0.0 1.0
   """
   input = solver_factory(molecule=xyz, basis='cc-pVDZ')

   # 1. singlet Ag state of H2 (neutral)
   state = input.state(charge=0, multiplicity=1, sym='ag') 

   # 2. create the HF object
   hf = HF(input, state=state)  

   # 3. run the computation
   hf.run()  

The output of this computation can be found in the ``output.dat`` file.
However, the results of this computation are also stored in the HF
object. For example, the HF energy can be accessed via
``hf.value('hf energy')``.

FCI and CASCI
-------------

Forte implements several solvers that diagonalize the Hamiltonian in a
(small) space of orbitals, including FCI, selected CI methods, and
generalized active space (GAS). To perform one of these computations
just pass an object that can provide molecular orbitals to an
``ActiveSpaceSolver`` object. For example, we can perform a CASCI
computation on the same molecule as above by passing the ``HF`` node to
an ``ActiveSpaceSolver`` node

.. code:: python

   from forte.solvers import solver_factory, HF, ActiveSpaceSolver

   xyz = """
   H 0.0 0.0 0.0
   H 0.0 0.0 1.0
   """
   input = solver_factory(molecule=xyz, basis='cc-pVDZ')

   state = input.state(charge=0, multiplicity=1, sym='ag') 

   # create the HF object
   hf = HF(input, state=state)  

   # specify the active space
   # we pass an array that specifies the number of active MOs per irrep
   # We use Cotton ordering, so this selects one MO from irrep 0 (Ag) and one from irrep 5 (B1u)
   mo_spaces = input.mo_spaces(active=[1, 0, 0, 0, 0, 1, 0, 0])

   # initialize a FCI solver and pass the HF object, the target electronic state, and the MO space information
   fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)

   # call run() on the FCI node
   fci.run()  

The CASCI energy can be accessed via the ``value`` function on the FCI
object. In this case it returns a vector containing the energy of all
the states computed:

.. code:: python

   fci.value('active space energy')[state] -> [-1.1083377195359851]

To compute two :math:`^1 A_{g}` states we can simply pass a dictionary
that maps states to number of desired solutions

.. code:: python

   fci = ActiveSpaceSolver(hf, type='FCI', states={state : 2}, mo_spaces=mo_spaces)

The energy of the two :math:`^1 A_{g}` states can still be retrieved
with the ``value`` function:

.. code:: python

   fci.value('active space energy')[state] -> [-1.1083377195359851, -0.2591786932627466]
