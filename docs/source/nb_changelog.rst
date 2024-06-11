Changelog
=========

Version 0.4.0
-------------

This is a major update that introduces several significant changes to
the way Forte runs computations.

-  Enables the Standard Workflow described in the Forte paper
-  Deprecate old CASSCF code
-  Renamed all ``CASSCF_*`` options to ``MCSCF_*``. This will require
   tweaking old input files.
-  By default, Forte’s standard workflow will run an MCSCF computation
   using the ``ACTIVE_SPACE_SOLVER`` provided in the input. This can be
   avoided by setting ``MCSCF_REFERENCE`` to ``False``.
-  By default, the MCSCF code will combine the ``FROZEN_DOCC`` with the
   ``RESTRICTED_DOCC`` orbitals and optimize them. The user can control
   this with the keyword ``MCSCF_FREEZE_CORE`` (set by default to
   ``False``).
-  After an MCSCF computation, ``MCSCF_FREEZE_CORE = FALSE`` will mix
   all inactive orbitals (frozen + restricted doubly occupied) in the
   semicanonicalization, while ``MCSCF_FREEZE_CORE = FALSE`` will
   semicanonicalize them separately.
-  Fixed printing of NOs which contained a bug (printed wrong orbital
   index)


