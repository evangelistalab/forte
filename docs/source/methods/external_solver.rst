.. _`sec:methods:external`:

Read external RDMs, Export integrals
====================================

.. sectionauthor:: Renke Huang

Forte implements the ``ExternalActiveSpaceMethod`` class to provide users with more flexilibity with the correlation computations.


Read external RDMs for correlation computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Forte external solver will check if there is a json file named ``rdms.json`` present in the current working directory. 
See `the test case <https://github.com/RenkeHuang/forte/blob/ext_doc/tests/methods/external_solver-1/rdms.json>`_ for the data format used in ``rdms.json``. 
If the file is present, the external solver will read the RDMs and compute the reference energy. 
Here is the simplest input for a DSRG-MRPT2 computation using external RDMs::
    
    import forte
    molecule h2{
        0 1
        H
        H 1 0.7
    }

    set {
        basis                cc-pVDZ
        scf_type             pk
        e_convergence        12
    }

    set forte {
        active_space_solver  external    # read rdms.json, generate as_ints.json
        active               [1, 0, 0, 0, 0, 1, 0, 0]
        restricted_docc      [0, 0, 0, 0, 0, 0, 0, 0]
        correlation_solver   dsrg-mrpt2
        dsrg_s               0.5
    }

    energy('forte')



Export integrals to disk
^^^^^^^^^^^^^^^^^^^^^^^^

Forte can export two types of integrals: 
1) active space integrals from an ``ActiveSpaceSolver`` (which define a bare Hamiltonian), 
2) DSRG-dressed integrals.
The minimal input for writing active space integrals to disk (save as ``as_ints.json``) is::
    
    import forte
    molecule h2{
        0 1
        H
        H 1 0.7
    }

    set {
        basis                cc-pVDZ
        scf_type             pk
        e_convergence        12
    }

    set forte {
        active_space_solver  external    # generate as_ints.json
        write_wfn            true        # save coeff.json
        active               [1, 0, 0, 0, 0, 1, 0, 0]
        restricted_docc      [0, 0, 0, 0, 0, 0, 0, 0]
    }

    energy('forte')

See `this example <https://github.com/RenkeHuang/forte/blob/ext_doc/tests/methods/external_solver-1/as_ints.json>`_ for the data and the data format used in ``as_ints.json``
Note that saving the ``wfn.Ca()`` to ``coeff.json`` is always recommended to avoid sign flips and confusions in defining orbitals.

For DSRG-dressed integrals, to enable the pipeline of the external active space solver and a correlation solver (computes the dressed integrals) that follows, 
two files ``rdms.json`` and ``coeff.json`` are required in the working directory. 
Then run the following input to export DSRG-dressed integrals to disk (save as ``dsrg_ints.json``)::
    
    import forte
    molecule h2{
        0 1
        H
        H 1 0.7
    }

    set {
        basis                cc-pVDZ
        scf_type             pk
        e_convergence        12
    }

    set forte {
        active_space_solver  external    # read rdms.json, generate as_ints.json
        read_wfn             true        # read coeff.json
        active               [1, 0, 0, 0, 0, 1, 0, 0]
        restricted_docc      [0, 0, 0, 0, 0, 0, 0, 0]
        correlation_solver   dsrg-mrpt2
        dsrg_s               0.5
        relax_ref            once        # generate dsrg_ints.json
        external_partial_relax true
    }

    energy('forte')

Note that the data format used in JSON file is the same for both types of integrals (``as_ints.json`` and ``dsrg_ints.json``).


Save RDMs to disk
^^^^^^^^^^^^^^^^^
Forte also provides the functionality of exporting the reference RDMs (from an ``ActiveSpaceSolver``) to disk (save as ``ref_rdms.json``). 
This file shares the same data format as ``rdms.json``, so you can simply rename it for the purpose of quick testing of the external solver. 
Note that the external solver is only an I/O interface and cannot compute the RDMs on its own. 
To generate RDMs, we need to use another active space solver that implements the RDM calculations (for example, ``FCI``, ``CAS``, ``ACI``). 
::

    import forte
    molecule h2{
        0 1
        H
        H 1 0.7
    }

    set {
        basis                cc-pVDZ
        scf_type             pk
        e_convergence        12
    }

    set forte {
        active_space_solver  fci         # use: fci, cas, aci
        write_rdm            true        # save ref_rdms.json
        write_wfn            true        # save coeff.json
        active               [1, 0, 0, 0, 0, 1, 0, 0]
        restricted_docc      [0, 0, 0, 0, 0, 0, 0, 0]
    }

    energy('forte')


Options
^^^^^^^

**WRITE_RDM**

Save RDMs to ``ref_rdms.json`` for external computations.

* Type: bool
* Default: False

**WRITE_WFN**

Save ``ref_wfn.Ca()`` to ``coeff.json`` for external computations.

* Type: bool
* Default: False

**READ_WFN**

Read ``ref_wfn.Ca()/ref_wfn.Cb()`` from ``coeff.json`` for external active space solver.

* Type: bool
* Default: False

**EXTERNAL_PARTIAL_RELAX**

Perform one relaxation step after building the DSRG effective Hamiltonian when using external active space solver.

* Type: bool
* Default: False

**EXT_RELAX_SOLVER**

Active space solver used in the relaxation when using external active space solver.

* Type: string
* Options: FCI, DETCI, CAS
* Default: FCI

**SAVE_SA_DSRG_INTS**

Save SA-DSRG dressed integrals to ``dsrg_ints.json``.

* Type: bool
* Default: False
