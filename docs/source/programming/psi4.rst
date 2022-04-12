.. _`sec:programming:psi`:

Psi4
====

.. sectionauthor:: Francesco A. Evangelista

Symmetry and the ``Dimension`` class
------------------------------------

In Forte, the irreducible representations (irreps) of Abelian point groups are represented using a zero-based integer.
The Cotton ordering of irreps is used, which can be found `here <http://www.psicode.org/psi4manual/master/psithonmol.html#symmetry>`_.
This ordering is convenient because the direct product of two irreps can be computed using the XOR operator.
For example, consider ${C_2v} symmetry. if ``ha``=A1 and ``hb`` , then their direct product can be computed as::

   // Assume C2v symmetry
   // Cotton ordering: [A1, A2, B1, B2]
   int ha = 1; // 1 = 01 = A2
   int hb = 3; // 3 = 11 = B2 
   int hab = ha ^ hb; // 10 = 2 = B1


The ``Vector`` and ``Matrix`` classes
-------------------------------------
