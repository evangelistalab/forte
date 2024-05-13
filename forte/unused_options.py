#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE DMRGSOLVER
#    //////////////////////////////////////////////////////////////

#    options.add_int("DMRG_WFN_MULTP", -1)

#    /*- The DMRGSCF wavefunction irrep uses the same conventions as PSI4.
#    How convenient :-).
#        Just to avoid confusion, it's copied here. It can also be found on
#        http://sebwouters.github.io/CheMPS2/classCheMPS2_1_1Irreps.html .

#        Symmetry Conventions        Irrep Number & Name
#        Group Number & Name         0 	1 	2 	3 	4 5
#    6
#    7
#        0: c1                       A
#        1: ci                       Ag 	Au
#        2: c2                       A 	B
#        3: cs                       A' 	A''
#        4: d2                       A 	B1 	B2 	B3
#        5: c2v                      A1 	A2 	B1 	B2
#        6: c2h                      Ag 	Bg 	Au 	Bu
#        7: d2h                      Ag 	B1g 	B2g 	B3g 	Au
#    B1u 	B2u 	B3u
#    -*/
#    options.add_int("DMRG_WFN_IRREP", -1)
#    /*- FrozenDocc for DMRG (frozen means restricted) -*/
#    options.add_list("DMRG_FROZEN_DOCC")

#    /*- The number of reduced renormalized basis states to be
#        retained during successive DMRG instructions -*/
#    options.add_list("DMRG_STATES")

#    /*- The energy convergence to stop an instruction
#        during successive DMRG instructions -*/
#    options.add_list("DMRG_ECONV")

#    /*- The maximum number of sweeps to stop an instruction
#        during successive DMRG instructions -*/
#    options.add_list("DMRG_MAXSWEEPS")
#    /*- The Davidson R tolerance (Wouters says this will cause RDms to be
#     * close to exact -*/
#    options.add_list("DMRG_DAVIDSON_RTOL")

#    /*- The noiseprefactors for successive DMRG instructions -*/
#    options.add_list("DMRG_NOISEPREFACTORS")

#    /*- Whether or not to print the correlation functions after the DMRG
#     * calculation -*/
#    options.add_bool("DMRG_PRINT_CORR", False)

#    /*- Whether or not to create intermediary MPS checkpoints -*/
#    options.add_bool("MPS_CHKPT", False)

#    /*- Convergence threshold for the gradient norm. -*/
#    options.add_double("DMRG_CONVERGENCE", 1e-6)

#    /*- Whether or not to store the unitary on disk (convenient for
#     * restarting). -*/
#    options.add_bool("DMRG_STORE_UNIT", True)

#    /*- Whether or not to use DIIS for DMRGSCF. -*/
#    options.add_bool("DMRG_DO_DIIS", False)

#    /*- When the update norm is smaller than this value DIIS starts. -*/
#    options.add_double("DMRG_DIIS_BRANCH", 1e-2)

#    /*- Whether or not to store the DIIS checkpoint on disk (convenient for
#     * restarting). -*/
#    options.add_bool("DMRG_STORE_DIIS", True)

#    /*- Maximum number of DMRGSCF iterations -*/
#    options.add_int("DMRGSCF_MAX_ITER", 100)

#    /*- Which root is targeted: 1 means ground state, 2 first excited state,
#     * etc. -*/
#    options.add_int("DMRG_WHICH_ROOT", 1)

#    /*- Whether or not to use state-averaging for roots >=2 with DMRG-SCF.
#     * -*/
#    options.add_bool("DMRG_AVG_STATES", True)

#    /*- Which active space to use for DMRGSCF calculations:
#           --> input with SCF rotations (INPUT)
#           --> natural orbitals (NO)
#           --> localized and ordered orbitals (LOC) -*/
#    options.add_str("DMRG_ACTIVE_SPACE", "INPUT", "INPUT NO LOC")

#    /*- Whether to start the active space localization process from a random
#     * unitary or the unit matrix. -*/
#    options.add_bool("DMRG_LOC_RANDOM", True)
#    /*-  -*/
#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE FULL CI QUANTUM MONTE-CARLO
#    //////////////////////////////////////////////////////////////
#    /*- The maximum value of beta -*/
#    options.add_double("START_NUM_WALKERS", 1000.0)
#    /*- Spawn excitation type -*/
#    options.add_str("SPAWN_TYPE", "RANDOM", "RANDOM ALL GROUND_AND_RANDOM")
#    /*- The number of walkers for shift -*/
#    options.add_double("SHIFT_NUM_WALKERS", 10000.0)
#    options.add_int("SHIFT_FREQ", 10)
#    options.add_double("SHIFT_DAMP", 0.1)
#    /*- Clone/Death scope -*/
#    options.add_bool("DEATH_PARENT_ONLY", False)
#    /*- initiator -*/
#    options.add_bool("USE_INITIATOR", False)
#    options.add_double("INITIATOR_NA", 3.0)
#    /*- Iterations in between variational estimation of the energy -*/
#    options.add_int("VAR_ENERGY_ESTIMATE_FREQ", 1000)
#    /*- Iterations in between printing information -*/
#    options.add_int("PRINT_FREQ", 100)

#    //////////////////////////////////////////////////////////////
#    ///
#    ///              OPTIONS FOR THE SRG MODULE
#    ///
#    //////////////////////////////////////////////////////////////
#    /*- The type of operator to use in the SRG transformation -*/
#    options.add_str("SRG_MODE", "DSRG", "DSRG CT")
#    /*- The type of operator to use in the SRG transformation -*/
#    options.add_str("SRG_OP", "UNITARY", "UNITARY CC")
#    /*- The flow generator to use in the SRG equations -*/
#    options.add_str("SRG_ETA", "WHITE", "WEGNER_BLOCK WHITE")
#    /*- The integrator used to propagate the SRG equations -*/

#    /*-  -*/

#    // --------------------------- SRG EXPERT OPTIONS
#    // ---------------------------

#    /*- Save Hbar? -*/
#    options.add_bool("SAVE_HBAR", False)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE PILOT FULL CI CODE
#    //////////////////////////////////////////////////////////////

#    /*- The density convergence criterion -*/
#    options.add_double("D_CONVERGENCE", 1.0e-8)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE V2RDM INTERFACE
#    //////////////////////////////////////////////////////////////
#    /*- Write Density Matrices or Cumulants to File -*/
#    options.add_str("WRITE_DENSITY_TYPE", "NONE", "NONE DENSITY CUMULANT")
#    /*- Average densities of different spins in V2RDM -*/
#    options.add_bool("AVG_DENS_SPIN", False)

#    //////////////////////////////////////////////////////////////
#    ///              OPTIONS FOR THE MR-DSRG MODULE
#    //////////////////////////////////////////////////////////////

#    /*- The code used to do CAS-CI.
#     *  - CAS   determinant based CI code
#     *  - FCI   string based FCI code
#     *  - DMRG  DMRG code
#     *  - V2RDM V2RDM interface -*/
#    options.add_str("CAS_TYPE", "FCI", "CAS FCI ACI DMRG V2RDM")
