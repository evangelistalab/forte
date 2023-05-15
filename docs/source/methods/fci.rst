## Spin-adapted FCI

In certain cases, convergence to a state with target multiplicity fails due to either variational collapse to a root of lower energy and different multiplicity or because no guess state can be found.

Forte implements within the determinant-based FCI code a procedure to perform the Davidson–Liu procedure in a basis of configuration state funcions (CSFs). CSFs are spin-adapted linear combinations of Slater determinants with a given orbital occupation pattern (electron configuration).

When expressed in the CSF basis a FCI state is given by:
$$

$$