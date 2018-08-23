.. _`sec:methods:ldsrg`:

Driven Similarity Renormalization Group
=======================================

.. codeauthor:: Francesco A. Evangelista, Chenyang Li, Kevin Hannon, Tianyuan Zhang
.. sectionauthor:: Tianyuan Zhang

Basic DSRG
^^^^^^^^^^

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
