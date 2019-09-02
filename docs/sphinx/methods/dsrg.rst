.. _`sec:methods:ldsrg`:

Driven Similarity Renormalization Group
=======================================

.. codeauthor:: Francesco A. Evangelista, Chenyang Li, Kevin Hannon, Tianyuan Zhang
.. sectionauthor:: Chenyang Li, Tianyuan Zhang

Basic DSRG
^^^^^^^^^^

Overview of DSRG Theory
+++++++++++++++++++++++

Driven similarity renormalization group (DSRG) is a numerically robust approach to consider 
dynamical (or weak) electron correlation effects. Specifically, the DSRG performs a *continuous* 
similarity transformation to the bare Born-Oppenheimer Hamiltonian `\hat{H}`,

.. math:: \bar{H}(s) = e^{-\hat{S}(s)} \hat{H} e^{\hat{S}(s)},

where `s` is the flow parameter defined in the range `[0, +\infty)` that controls the transformation, vaguely speaking.
The operator `\hat{S}` can be any operator in general.
For example, if `\hat{S} = \hat{T}` is the cluster substitution operator, the DSRG `\bar{H}(s)` 
is identical to coupled-cluster (CC) similarity transformed Hamiltonian except for the `s` 
dependence. See Table I for different flavours of `\hat{S}`.

Table I. Connections of DSRG to CC theories when using different types of `\hat{S}`.:

+------------+------------------------------------+----------------+
| `\hat{S}`  |             Explanation            |   CC Theories  |
+============+====================================+================+
| `\hat{T}`  |          cluster operator          | traditional CC |
+------------+------------------------------------+----------------+
| `\hat{A}`  | `\hat{A} = \hat{T} - \hat{T}^\dag` | unitary CC, CT |
+------------+------------------------------------+----------------+
| `\hat{G}`  |          general operator          | generalized CC |
+------------+------------------------------------+----------------+

In the current implementation, we choose the **anti-hermitian** parametrization, i.e., 
`\hat{S} = \hat{A}`.

The DSRG transformed Hamiltonian contains many-body (> 2-body) interactions in general.
We can express it as

.. math:: \bar{H} = \bar{h}_0 + \bar{h}^{p}_{q} \{ a^{q}_{p} \} + \frac{1}{4} \bar{h}^{pq}_{rs} \{ a^{rs}_{pq} \} + \frac{1}{36} \bar{h}^{pqr}_{stu} \{ a^{stu}_{pqr} \} + ...

where `a^{pq...}_{rs...} = a_{p}^\dag a_{q}^\dag \dots a_s a_r` is a string of creation and annihilation operators
and `\{\cdot\}` represents normal-ordered operators. In particular, we use Mukherjee-Kutzelnigg normal ordering
[see J. Chem. Phys. 107, 432 (1997)]. Here we also assume summations over repeated indices for brevity.
Also note that `\bar{h}_0` is the energy dressed by dynamical correlation effects.

In DSRG, we require the off-diagonal components of `\bar{H}` gradually go to zero (from `\hat{H}`) as `s` grows (from 0).
By off-diagonal components, we mean `\bar{h}^{ij\dots}_{ab\dots}` and `\bar{h}^{ab\dots}_{ij\dots}` where `i,j,\dots`
indicates hole orbitals and `a,b,\dots` labels particle orbitals.
There are in principle infinite numbers of ways to achieve this requirement.
The current implementation chooses the following parametrization,

.. math:: \bar{h}^{ij\dots}_{ab\dots} = [\bar{h}^{ij\dots}_{ab\dots} + \Delta^{ij\dots}_{ab\dots} t^{ij\dots}_{ab\dots}] e^{-s(\Delta^{ij\dots}_{ab\dots})^2},

where `\Delta^{ij\dots}_{ab\dots} = \epsilon_{i} + \epsilon_{j} + \dots - \epsilon_{a} - \epsilon_{b} - \dots` is
the Moller-Plesset denominator defined by orbital energies `\epsilon_{p}` and `t^{ij\dots}_{ab\dots}` is the cluster amplitudes.
This equation is called the DSRG flow equation, which suggests a way how the off-diagonal Hamiltonian components are zeroed.
We can now solve for the cluster amplitudes since `\bar{H}` is a function of `\hat{T}` using the Baker–Campbell–Hausdorff (BCH) formula.

Since we choose `\hat{S} = \hat{A}`, the corresponding BCH expansion is thus non-terminating.
Approximations has to be introduced and different treatments to `\bar{H}` leads to various levels of DSRG theories.
Generally, we can treat it either in a perturbative or non-perturbative manner.
For non-perturbative theories, the **only** widely tested scheme so far is the recursive single commutator (RSC) approach,
where every single commutator is truncated to contain at most two-body contributions for a nested commutator.
For example, a doubly nested commutator is computed as

.. math:: \frac{1}{2} [[\hat{H}, \hat{A}], \hat{A}] \approx \frac{1}{2} [[\hat{H}, \hat{A}]_{1,2}, \hat{A}]_{0,1,2},

where 0, 1, 2 indicate scalar, 1-body, and 2-body contributions.
We term the DSRG method that uses RSC as LDSRG(2).

Alternatively, we can perform a perturbative analysis on the **approximated** BCH equation of `\bar{H}` and obtain
various DSRG perturbation theories (e.g., 2nd-order or 3rd-order).
Note we use the RSC approximated BCH equation for computational cost considerations.
As such, the implemented DSRG-PT3 is **not** a complete PT3 but a companion PT3 of the LDSRG(2) method.

To conclude this subsection, we discuss the computational cost and current implementation limit.
These aspects are summarized in Table II.

Table II. Cost of the various implemented DSRG methods:

+----------+--------------------+----------------------------+---------------------+
|  Method  | Computational Cost | System Size (full 2e-ints) | System Size (DF/CD) |
+==========+====================+============================+=====================+
|    PT2   |   one-shot `N^5`   | `~250`                     | `~2000`             |
+----------+--------------------+----------------------------+---------------------+
|    PT3   |   one-shot `N^6`   | `~250`                     | `~700`              |
+----------+--------------------+----------------------------+---------------------+
| LDSRG(2) |   iterative `N^6`  | `~200`                     | `~500`              |
+----------+--------------------+----------------------------+---------------------+


Input Examples
++++++++++++++


DSRG Options
~~~~~~~~~~~~

**CORR_LEVEL**

Correlation level of MR-DSRG.

* Type: string
* Options: PT2, PT3, LDSRG2, LDSRG2_QC, LSRG2, SRG_PT2, QDSRG2, LDSRG2_P3, QDSRG2_P3
* Default: PT2

**DSRG_S**

The value of the flow parameter :math:`s`.

* Type: double
* Default: 1.0e10


MR-LDSRG(2)
^^^^^^^^^^^

MR-LDSRG(2) Options
~~~~~~~~~~~~~~~~~~~

**DSRG_MAXITER**

Max iterations for MR-DSRG amplitudes update.

* Type: int
* Default: 50

**RELAX_REF**

Relax the reference for MR-DSRG.

* Type: string
* Options: NONE, ONCE, TWICE, ITERATE
* Default: NONE

**DSRG_HBAR_SEQ**

Apply the sequential transformation algorithm in evaluating the transformed Hamiltonian :math:`\bar{H}(s)`, i.e.,

.. math:: \bar{H}(s) = e^{-\hat{A}_n(s)} \cdots e^{-\hat{A}_2(s)} e^{-\hat{A}_1(s)} \hat{H} e^{\hat{A}_1(s)} e^{\hat{A}_2(s)} \cdots e^{\hat{A}_n(s)}.

* Type: bool
* Default: false

**DSRG_NIVO**

Apply non-interacting virtual orbital (NIVO) approximation in evaluating the transformed Hamiltonian.

* Type: bool
* Default: false
