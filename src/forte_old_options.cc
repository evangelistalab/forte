#include "psi4/liboptions/liboptions.h"

#include "aci/aci.h"
#include "fci/fci_solver.h"
#include "fci/fci.h"
#include "integrals/integrals.h"
#include "pci/pci.h"

namespace psi {
namespace forte {

void forte_old_options(Options& options) {

    /*- MODULEDESCRIPTION Forte */

    /*- SUBSECTION Job Type */

    /*- Compute natural orbitals using MP2 -*/
    options.add_bool("MP2_NOS", false);
    /*- View the natural orbitals with their symmetry information -*/
    options.add_bool("NAT_ORBS_PRINT", false);
    /*- Use Natural Orbitals to suggest active space -*/
    options.add_bool("NAT_ACT", false);
    options.add_bool("MOLDEN_WRITE_FORTE", false);

    // Natural Orbital selection criteria.  Used to fine tune how many
    // active orbitals there are

    /*- Typically, a occupied orbital with a NO occupation of <0.98 is
     * considered active -*/
    options.add_double("OCC_NATURAL", 0.98);
    /*- Typically, a virtual orbital with a NO occupation of > 0.02 is
     * considered active -*/
    options.add_double("VIRT_NATURAL", 0.02);

    /*- The amount of information printed
        to the output file -*/
    options.add_int("PRINT", 0);
    /*- Print summary of memory -*/
    options.add_bool("MEMORY_SUMMARY", false);

    /*- The symmetry of the electronic state. (zero based) -*/
    options.add_int("ROOT_SYM", 0);

    /*- The multiplicity (2S + 1) of the electronic state.
     *  For example, 1 = singlet, 2 = doublet, 3 = triplet, ...
     *  If a value is provided it overrides the multiplicity
     *  of the SCF solution. -*/
    options.add_int("MULTIPLICITY", 0);
    /*- The number of trials for root for FCI -*/
    options.add_int("NTRIAL_PER_ROOT", 10);

    /*- The charge of the molecule.  If a value is provided
        it overrides the charge of the SCF solution. -*/
    options.add_int("CHARGE", 0);

    /*- The minimum excitation level (Default value: 0) -*/
    options.add_int("MIN_EXC_LEVEL", 0);

    /*- The maximum excitation level (Default value: 0 = number of
     * electrons) -*/
    options.add_int("MAX_EXC_LEVEL", 0);

    /*- Number of frozen occupied orbitals per irrep (in Cotton order) -*/
    options.add("FROZEN_DOCC", new ArrayType());

    /*- Number of restricted doubly occupied orbitals per irrep (in Cotton
     * order) -*/
    options.add("RESTRICTED_DOCC", new ArrayType());

    /*- Number of active orbitals per irrep (in Cotton order) -*/
    options.add("ACTIVE", new ArrayType());

    /*- Number of restricted unoccupied orbitals per irrep (in Cotton order)
     * -*/
    options.add("RESTRICTED_UOCC", new ArrayType());

    /*- Number of frozen unoccupied orbitals per irrep (in Cotton order) -*/
    options.add("FROZEN_UOCC", new ArrayType());
    /*- Molecular orbitals to swap -
     *  Swap mo_1 with mo_2 in irrep symmetry
     *  Swap mo_3 with mo_4 in irrep symmetry
     *  Format: [irrep, mo_1, mo_2, irrep, mo_3, mo_4]
     *          Irrep and MO indices are 1-based (NOT 0-based)!
    -*/
    options.add("ROTATE_MOS", new ArrayType());

    /*- The algorithm used to screen the determinant
     *  - DENOMINATORS uses the MP denominators to screen strings
     *  - SINGLES generates the space by a series of single excitations -*/
    options.add_str("EXPLORER_ALGORITHM", "DENOMINATORS", "DENOMINATORS SINGLES");

    /*- The energy threshold for the determinant energy in Hartree -*/
    options.add_double("DET_THRESHOLD", 1.0);

    /*- The energy threshold for the MP denominators energy in Hartree -*/
    options.add_double("DEN_THRESHOLD", 1.5);

    /*- The criteria used to screen the strings -*/
    options.add_str("SCREENING_TYPE", "MP", "MP DET");

    // Options for the diagonalization of the Hamiltonian //
    /*- Determines if this job will compute the energy -*/
    options.add_bool("COMPUTE_ENERGY", true);

    /*- The form of the Hamiltonian matrix.
     *  - FIXED diagonalizes a matrix of fixed dimension
     *  - SMOOTH forms a matrix with smoothed matrix elements -*/
    options.add_str("H_TYPE", "FIXED_ENERGY", "FIXED_ENERGY FIXED_SIZE");

    /*- Determines if this job will compute the energy -*/
    options.add_str("ENERGY_TYPE", "FULL", "FULL SELECTED LOWDIN SPARSE RENORMALIZE "
                                           "RENORMALIZE_FIXED LMRCISD LMRCIS IMRCISD "
                                           "IMRCISD_SPARSE LMRCISD_SPARSE LMRCIS_SPARSE "
                                           "FACTORIZED_CI");

    /*- The form of the Hamiltonian matrix.
     *  - FIXED diagonalizes a matrix of fixed dimension
     *  - SMOOTH forms a matrix with smoothed matrix elements -*/

    //    options.add_int("IMRCISD_TEST_SIZE", 0);
    //    options.add_int("IMRCISD_SIZE", 0);

    /*- The number of determinants used to build the Hamiltonian -*/
    options.add_int("NDETS", 100);

    /*- The maximum dimension of the Hamiltonian -*/
    options.add_int("MAX_NDETS", 1000000);

    /*- The energy threshold for the model space -*/
    options.add_double("SPACE_M_THRESHOLD", 1000.0);

    /*- The energy threshold for the intermdiate space -*/
    options.add_double("SPACE_I_THRESHOLD", 1000.0);

    /*- The energy threshold for the intermdiate space -*/
    options.add_double("T2_THRESHOLD", 0.000001);

    /*- The number of steps used in the renormalized Lambda CI -*/
    options.add_int("RENORMALIZATION_STEPS", 10);

    /*- The maximum number of determinant in the fixed-size renormalized
     * Lambda CI -*/
    options.add_int("REN_MAX_NDETS", 1000);

    /*- The energy threshold for smoothing the Hamiltonian.
     *  Determinants with energy < DET_THRESHOLD - SMO_THRESHOLD will be
     * included in H
     *  Determinants with DET_THRESHOLD - SMO_THRESHOLD < energy <
     * DET_THRESHOLD will be included in H but smoothed
     *  Determinants with energy > DET_THRESHOLD will not be included in H
     * -*/
    options.add_double("SMO_THRESHOLD", 0.0);

    /*- The method used to smooth the Hamiltonian -*/
    options.add_bool("SMOOTH", false);

    /*- The method used to smooth the Hamiltonian -*/
    options.add_bool("SELECT", false);

    /*- The diagonalization method -*/
    options.add_str("DIAG_ALGORITHM", "SOLVER", "DAVIDSON FULL DAVIDSONLIST SPARSE SOLVER");

    options.add_str("SIGMA_BUILD_TYPE", "SPARSE", "SPARSE HZ MMULT");

    /*- Force the diagonalization procedure?  -*/
    options.add_bool("FORCE_DIAG_METHOD", false);

    /*- The energy convergence criterion -*/
    options.add_double("E_CONVERGENCE", 1.0e-8);

    /*- The energy relaxation convergence criterion -*/
    options.add_double("RELAX_E_CONVERGENCE", 1.0e-8);

    /*- The number of roots computed -*/
    options.add_int("NROOT", 1);

    /*- The root selected for state-specific computations -*/
    options.add_int("ROOT", 0);

    // Options for the Cartographer class //
    /*- Density of determinants format -*/
    options.add_str("DOD_FORMAT", "HISTOGRAM", "GAUSSIAN HISTOGRAM");
    /*- Number of bins used to form the DOD plot -*/
    options.add_int("DOD_BINS", 2000);
    /*- Width of the DOD Gaussian/histogram.  Default 0.02 Hartree ~ 0.5 eV
     * -*/
    options.add_double("DOD_BIN_WIDTH", 0.05);
    /*- Write an output file? -*/
    options.add_bool("DETTOUR_WRITE_FILE", false);
    /*- Write the determinant occupation? -*/
    options.add_bool("WRITE_OCCUPATION", true);
    /*- Write the determinant energy? -*/
    options.add_bool("WRITE_DET_ENERGY", true);
    /*- Write the denominator energy? -*/
    options.add_bool("WRITE_DEN_ENERGY", false);
    /*- Write the excitation level? -*/
    options.add_bool("WRITE_EXC_LEVEL", false);
    /*- Write information only for a given excitation level.
        0 (default) means print all -*/
    options.add_int("RESTRICT_EXCITATION", 0);
    /*- The energy buffer for building the Hamiltonian matrix in Hartree -*/
    options.add_double("H_BUFFER", 0.0);

    /*- The maximum number of iterations -*/
    options.add_int("MAXITER", 100);

    /*- Use localized basis? -*/
    options.add_bool("LOCALIZE", false);
    /*- Use fully localized basis? -*/
    options.add_bool("FULLY_LOCALIZE", false);
    /*- Type of localization -*/
    options.add_str("LOCALIZE_TYPE", "PIPEK_MEZEY", "BOYS");

    /*- Number of orbitals for CI guess  -*/
    options.add_int("ACTIVE_GUESS_SIZE", 1000);

    // Options for the Genetic Algorithm CI //
    /*- The size of the population -*/
    //    options.add_int("NPOP", 100);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR ALTERNATIVES FOR CASSCF ORBITALS
    //////////////////////////////////////////////////////////////
    /*- What type of alternative CASSCF Orbitals do you want -*/
    options.add_str("ALTERNATIVE_CASSCF", "NONE", "IVO FTHF NONE");
    options.add_double("TEMPERATURE", 50000);

    /*- Print the NOs -*/
    options.add_bool("PRINT_NO", false);
    /*- The number of trial guess vectors to generate per root -*/
    options.add_int("NTRIAL_PER_ROOT", 10);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE DAVIDSON-LIU SOLVER
    //////////////////////////////////////////////////////////////

    /*- The maximum number of iterations -*/
    options.add_int("DL_MAXITER", 100);
    /*- The number of trial vector to retain after collapsing -*/
    options.add_int("DL_COLLAPSE_PER_ROOT", 2);
    /*- The maxim number of trial vectors -*/
    options.add_int("DL_SUBSPACE_PER_ROOT", 8);
    /*- The maxim number of trial vectors -*/
    options.add_int("SIGMA_VECTOR_MAX_MEMORY", 10000000);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE CASSCF CODE
    //////////////////////////////////////////////////////////////
    /* - Run a FCI followed by CASSCF computation -*/
    options.add_bool("CASSCF_REFERENCE", false);
    /* - The number of iterations for CASSCF -*/
    options.add_int("CASSCF_ITERATIONS", 30);
    /* - The convergence for the gradient for casscf -*/
    options.add_double("CASSCF_G_CONVERGENCE", 1e-4);
    /* - The convergence of the energy for CASSCF -*/
    options.add_double("CASSCF_E_CONVERGENCE", 1e-6);
    /* - Debug printing for CASSCF -*/
    options.add_bool("CASSCF_DEBUG_PRINTING", false);
    /* - Multiplicity for the CASSCF solution (if different from
     multiplicity)
     You should not use this if you are interested in having a CASSCF
     solution with the same multiplicitity as the DSRG-MRPT2- */

    options.add_int("CASSCF_MULTIPLICITY", 0);
    /*- A complete SOSCF ie Form full Hessian -*/
    options.add_bool("CASSCF_SOSCF", false);
    /*- Ignore frozen core option and optimize orbitals -*/
    options.add_bool("OPTIMIZE_FROZEN_CORE", false);
    /*- CASSCF MAXIMUM VALUE HESSIAN -*/
    options.add_double("CASSCF_MAX_ROTATION", 0.5);
    /*- DO SCALE THE HESSIAN -*/
    options.add_bool("CASSCF_SCALE_ROTATION", true);
    /*- Use JK builder for restricted docc (EXPERT) -*/
    options.add_bool("RESTRICTED_DOCC_JK", true);
    /*- Orbital rotation algorithm -*/
    options.add_str("ORB_ROTATION_ALGORITHM", "DIAGONAL", "DIAGONAL AUGMENTED_HESSIAN");

    /*- DIIS Options -*/
    options.add_bool("CASSCF_DO_DIIS", true);
    /// The number of rotation parameters to extrapolate with
    options.add_int("CASSCF_DIIS_MAX_VEC", 8);
    /// When to start the DIIS iterations (will make this automatic)
    options.add_int("CASSCF_DIIS_START", 3);
    /// How often to do DIIS extrapolation
    options.add_int("CASSCF_DIIS_FREQ", 1);
    /// When the norm of the orbital gradient is below this value, do diis
    options.add_double("CASSCF_DIIS_NORM", 1e-3);
    /// Do a CAS step for every CASSCF_CI_FREQ
    options.add_bool("CASSCF_CI_STEP", false);
    /// How often should you do the CI_FREQ
    options.add_int("CASSCF_CI_FREQ", 1);
    /// When to start skipping CI steps
    options.add_int("CASSCF_CI_STEP_START", -1);

    //////////////////////////////////////////////////////////////
    /// OPTIONS FOR STATE-AVERAGE CASCI/CASSCF
    //////////////////////////////////////////////////////////////
    /*- An array of states [[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...] -*/
    options.add("AVG_STATE", new ArrayType());
    /*- An array of weights [[w1_1, w1_2, ..., w1_n], [w2_1, w2_2, ..., w2_n], ...] -*/
    options.add("AVG_WEIGHT", new ArrayType());
    /*- Monitor the CAS-CI solutions through iterations -*/
    options.add_bool("MONITOR_SA_SOLUTION", false);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE DMRGSOLVER
    //////////////////////////////////////////////////////////////

    options.add_int("DMRG_WFN_MULTP", -1);

    /*- The DMRGSCF wavefunction irrep uses the same conventions as PSI4.
    How convenient :-).
        Just to avoid confusion, it's copied here. It can also be found on
        http://sebwouters.github.io/CheMPS2/classCheMPS2_1_1Irreps.html .

        Symmetry Conventions        Irrep Number & Name
        Group Number & Name         0 	1 	2 	3 	4 5
    6
    7
        0: c1                       A
        1: ci                       Ag 	Au
        2: c2                       A 	B
        3: cs                       A' 	A''
        4: d2                       A 	B1 	B2 	B3
        5: c2v                      A1 	A2 	B1 	B2
        6: c2h                      Ag 	Bg 	Au 	Bu
        7: d2h                      Ag 	B1g 	B2g 	B3g 	Au
    B1u 	B2u 	B3u
    -*/
    options.add_int("DMRG_WFN_IRREP", -1);
    /*- FrozenDocc for DMRG (frozen means restricted) -*/
    options.add_array("DMRG_FROZEN_DOCC");

    /*- The number of reduced renormalized basis states to be
        retained during successive DMRG instructions -*/
    options.add_array("DMRG_STATES");

    /*- The energy convergence to stop an instruction
        during successive DMRG instructions -*/
    options.add_array("DMRG_ECONV");

    /*- The maximum number of sweeps to stop an instruction
        during successive DMRG instructions -*/
    options.add_array("DMRG_MAXSWEEPS");
    /*- The Davidson R tolerance (Wouters says this will cause RDms to be
     * close to exact -*/
    options.add_array("DMRG_DAVIDSON_RTOL");

    /*- The noiseprefactors for successive DMRG instructions -*/
    options.add_array("DMRG_NOISEPREFACTORS");

    /*- Whether or not to print the correlation functions after the DMRG
     * calculation -*/
    options.add_bool("DMRG_PRINT_CORR", false);

    /*- Whether or not to create intermediary MPS checkpoints -*/
    options.add_bool("MPS_CHKPT", false);

    /*- Convergence threshold for the gradient norm. -*/
    options.add_double("DMRG_CONVERGENCE", 1e-6);

    /*- Whether or not to store the unitary on disk (convenient for
     * restarting). -*/
    options.add_bool("DMRG_STORE_UNIT", true);

    /*- Whether or not to use DIIS for DMRGSCF. -*/
    options.add_bool("DMRG_DO_DIIS", false);

    /*- When the update norm is smaller than this value DIIS starts. -*/
    options.add_double("DMRG_DIIS_BRANCH", 1e-2);

    /*- Whether or not to store the DIIS checkpoint on disk (convenient for
     * restarting). -*/
    options.add_bool("DMRG_STORE_DIIS", true);

    /*- Maximum number of DMRGSCF iterations -*/
    options.add_int("DMRGSCF_MAX_ITER", 100);

    /*- Which root is targeted: 1 means ground state, 2 first excited state,
     * etc. -*/
    options.add_int("DMRG_WHICH_ROOT", 1);

    /*- Whether or not to use state-averaging for roots >=2 with DMRG-SCF.
     * -*/
    options.add_bool("DMRG_AVG_STATES", true);

    /*- Which active space to use for DMRGSCF calculations:
           --> input with SCF rotations (INPUT);
           --> natural orbitals (NO);
           --> localized and ordered orbitals (LOC) -*/
    options.add_str("DMRG_ACTIVE_SPACE", "INPUT", "INPUT NO LOC");

    /*- Whether to start the active space localization process from a random
     * unitary or the unit matrix. -*/
    options.add_bool("DMRG_LOC_RANDOM", true);
    /*- Use the older DMRGSCF algorithm -*/
    options.add_bool("USE_DMRGSCF", false);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE FULL CI QUANTUM MONTE-CARLO
    //////////////////////////////////////////////////////////////
    /*- The maximum value of beta -*/
    options.add_double("START_NUM_WALKERS", 1000.0);
    /*- Spawn excitation type -*/
    options.add_str("SPAWN_TYPE", "RANDOM", "RANDOM ALL GROUND_AND_RANDOM");
    /*- The number of walkers for shift -*/
    options.add_double("SHIFT_NUM_WALKERS", 10000.0);
    options.add_int("SHIFT_FREQ", 10);
    options.add_double("SHIFT_DAMP", 0.1);
    /*- Clone/Death scope -*/
    options.add_bool("DEATH_PARENT_ONLY", false);
    /*- initiator -*/
    options.add_bool("USE_INITIATOR", false);
    options.add_double("INITIATOR_NA", 3.0);
    /*- Iterations in between variational estimation of the energy -*/
    options.add_int("VAR_ENERGY_ESTIMATE_FREQ", 1000);
    /*- Iterations in between printing information -*/
    options.add_int("PRINT_FREQ", 100);

    //////////////////////////////////////////////////////////////
    ///
    ///              OPTIONS FOR THE SRG MODULE
    ///
    //////////////////////////////////////////////////////////////
    /*- The type of operator to use in the SRG transformation -*/
    options.add_str("SRG_MODE", "DSRG", "DSRG CT");
    /*- The type of operator to use in the SRG transformation -*/
    options.add_str("SRG_OP", "UNITARY", "UNITARY CC");
    /*- The flow generator to use in the SRG equations -*/
    options.add_str("SRG_ETA", "WHITE", "WEGNER_BLOCK WHITE");
    /*- The integrator used to propagate the SRG equations -*/
    options.add_str("SRG_ODEINT", "FEHLBERG78", "DOPRI5 CASHKARP FEHLBERG78");
    /*- The end value of the integration parameter s -*/
    options.add_double("SRG_SMAX", 10.0);
    /*- The end value of the integration parameter s -*/
    options.add_double("DSRG_S", 1.0e10);
    /*- The end value of the integration parameter s -*/
    options.add_double("DSRG_POWER", 2.0);

    // --------------------------- SRG EXPERT OPTIONS
    // ---------------------------

    /*- The initial time step used by the ode solver -*/
    options.add_double("SRG_DT", 0.001);
    /*- The absolute error tollerance for the ode solver -*/
    options.add_double("SRG_ODEINT_ABSERR", 1.0e-12);
    /*- The absolute error tollerance for the ode solver -*/
    options.add_double("SRG_ODEINT_RELERR", 1.0e-12);
    /*- Select a modified commutator -*/
    options.add_str("SRG_COMM", "STANDARD", "STANDARD FO FO2");
    /*- The maximum number of commutators in the recursive single commutator
     * approximation -*/
    options.add_int("DSRG_RSC_NCOMM", 20);
    /*- The treshold for terminating the RSC approximation -*/
    options.add_double("SRG_RSC_THRESHOLD", 1.0e-12);
    /*- Save Hbar? -*/
    options.add_bool("SAVE_HBAR", false);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE PILOT FULL CI CODE
    //////////////////////////////////////////////////////////////
    /*- Semicanonicalize orbitals -*/
    options.add_bool("SEMI_CANONICAL", true);
    /*- Two-particle density cumulant -*/
    options.add_str("TWOPDC", "MK", "MK ZERO");
    /*- Three-particle density cumulant -*/
    options.add_str("THREEPDC", "MK", "MK MK_DECOMP ZERO");
    /*- Number of roots per irrep (in Cotton order) -*/
    options.add("NROOTPI", new ArrayType());
    /*- The density convergence criterion -*/
    options.add_double("D_CONVERGENCE", 1.0e-8);

    //////////////////////////////////////////////////////////////
    ///         OPTIONS FOR THE V2RDM INTERFACE
    //////////////////////////////////////////////////////////////
    /*- Write Density Matrices or Cumulants to File -*/
    options.add_str("WRITE_DENSITY_TYPE", "NONE", "NONE DENSITY CUMULANT");
    /*- Average densities of different spins in V2RDM -*/
    options.add_bool("AVG_DENS_SPIN", false);

    //////////////////////////////////////////////////////////////
    ///              OPTIONS FOR THE MR-DSRG MODULE
    //////////////////////////////////////////////////////////////

    /*- The code used to do CAS-CI.
     *  - CAS   determinant based CI code
     *  - FCI   string based FCI code
     *  - DMRG  DMRG code
     *  - V2RDM V2RDM interface -*/
    options.add_str("CAS_TYPE", "FCI", "CAS FCI ACI DMRG V2RDM");
}
}
} // End Namespaces
