.. _`sec:multistate`:

Specifying calculations of multiple states
==========================================

.. sectionauthor:: Francesco A. Evangelista


Requesting multiple solutions of a given spin and symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Codes that support excited states take the additional option ``NROOT``,
which can be used to specify the number of solutions (roots) of the
charge, multiplicity, and symmetry specified by the user.
This type of computation

Assuming a :math:`C_{2v}` molecular point group, the following example is for
an input to compute three state of symmetry :math:`^{4}A_{2}` for a neutral
molecule::

    set forte {
      charge       0 # <-- neutral
      multiplicity 4 # <-- quartet
      root_sym     1 # <-- A_2
      nroot        3 # <-- three solutions
    }


Requesting multiple solutions of different spin and symmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For certain type of multistate computations (e.g., state-averaged CASSCF)
one may want to compute solutions of different spin and symmetry

The simplest way to do so is by specifying the ``AVG_STATE`` option to
defined groups of electronic states to compute.
This option is passed as a list of triplets of numbers
``[[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...]``,
where ``irrep``, ``multi``, and ``nstates`` specify the irrep, multip
licity,
and the number of states of each type requested.

For example, for a molecule with :math:`C_{2v}` point group symmetry,
the following input requests four :math:`^{3}B_{1}` states and
two :math:`^{5}B_{2}` states::

    set forte {
      avg_state [[2,3,4],[3,5,2]] # <-- [(B1, triplet, 4 states), (B2,quintet,2 states)]
    } 

When ``AVG_STATE`` is specified, each state is assigned a weight, which 
by default is :math:`1/N` where :math:`N` is the total number of states
computed.
The weights of all the states can also be indicated with the ``AVG_WEIGHT``
option. This option is a list of lists of numbers that indicate the weight of
each state in the group of states defined via ``AVG_STATE``.
This option takes the format ``[[w1_1, w1_2, ..., w1_l],
" [w2_1, w2_2, ..., w2_m], ...]``

Suppose we want to do a computation on a singlet and triplet :math:`A_{1}` state,
and assign thest states weights 1/4 and 3/4. This computation can be specified by
the input::

set forte {
  avg_state [[0,1,1],[0,3,1]]
  avg_weight [[0.25],[0.75]]
} 

Multistate GAS calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multistate computations using a GAS partitioning can be used to generate
even more nuanced electronic states.
When the electronic states are specified via the ``AVG_STATE`` option,
one can indicate states with different GAS occupations by setting the
``GASNMIN`` and ``GASNMAX`` options. For multistate computations,
these are lists that specify the minimum and maxium occupation of each
GAS space for each of the group of electronic states.

For example, the test case ``tests/methods/gasci-2`` shows how to compute
two electronic states of the water molecule of :math:`^1A_1` symmetry.
These two states use different occupation restrictions. Specifically,
the O 1s-like orbital (:math:`1a_1`) has maximum occupation of 2 and 1
in the two electronic states::

    set fort {
      gas1    [1,0,0,0]
      gas2    [3,0,1,2]
      gas1min [0,0]
      gas1max [2,1]
      avg_state [[0,1,1],[0,1,1]] # 2 states of singlet A_1 symmetry and different GAS
    }

While the first state is representative of the ground state of water,
the second state corresponds to a core-excited state.

