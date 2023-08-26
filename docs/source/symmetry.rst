Point group symmetry
--------------------

Forte takes advantage of symmetry, so it important to specify both the
symmetry of the target electronic state and the orbital spaces that
define a computation (see below). Forte supports only Abelian groups
(:math:`C_1`, :math:`C_s`, :math:`C_i`, :math:`C_2`, :math:`C_{2h}`,
:math:`C_{2v}`, :math:`D_2`, :math:`D_{2h}`). If a molecule has
non-Abelian point group symmetry, the largest Abelian subgroup will be
used. For a given group, the irreducible representations (irrep) are
arranged according to Cotton’s book (*Chemical Applications of Group
Theory*). This ordering is reproduced in the following table and is the
same as used in Psi4:

.. list-table::
   :widths: 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - Point group
     - Irrep 0
     - Irrep 1
     - Irrep 2
     - Irrep 3
     - Irrep 4
     - Irrep 5
     - Irrep 6
     - Irrep 7
   * - :math:`C_1`
     - :math:`A`
     - 
     - 
     - 
     - 
     - 
     - 
     - 
   * - :math:`C_s`
     - :math:`A'`
     - :math:`A''`
     - 
     - 
     - 
     - 
     - 
     - 
   * - :math:`C_i`
     - :math:`A_{g}`
     - :math:`A_{u}`
     - 
     - 
     - 
     - 
     - 
     - 
   * - :math:`C_2`
     - :math:`A`
     - :math:`B`
     - 
     - 
     - 
     - 
     - 
     - 
   * - :math:`C_{2h}`
     - :math:`A_{g}`
     - :math:`B_{g}`
     - :math:`A_{u}`
     - :math:`B_{u}`
     - 
     - 
     - 
     - 
   * - :math:`C_{2v}`
     - :math:`A_{1}`
     - :math:`B_{1}`
     - :math:`A_{2}`
     - :math:`B_{2}`
     - 
     - 
     - 
     - 
   * - :math:`D_2`
     - :math:`A`
     - :math:`B_{1}`
     - :math:`B_{2}`
     - :math:`B_{3}`
     - 
     - 
     - 
     - 
   * - :math:`D_{2h}`
     - :math:`A_{g}`
     - :math:`B_{1g}`
     - :math:`B_{2g}`
     - :math:`B_{3g}`
     - :math:`A_{u}`
     - :math:`B_{1u}`
     - :math:`B_{2u}`
     - :math:`B_{3u}`

By default, Forte targets a total symmetric state (e.g., :math:`A_1`,
:math:`A_{g}`, …). To specify a state with a different irreducible
representation (irrep), provide the ``ROOT_SYM`` option. This option
takes an integer argument that indicates the irrep in Cotton’s ordering.
