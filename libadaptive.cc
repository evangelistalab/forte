#include <cmath>
#include <memory>

#include <ambit/tensor.h>

#include "psi4-dec.h"
#include "psifiles.h"
#include <libplugin/plugin.h>
#include <libdpd/dpd.h>
#include <libpsio/psio.hpp>
#include <libtrans/integraltransform.h>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include "multidimensional_arrays.h"

#include "helpers.h"

#include "mp2_nos.h"
#include "adaptive-ci.h"
#include "ex-aci.h"
#include "adaptive_pici.h"
#include "fast_apici.h"
#include "lambda-ci.h"
#include "fcimc.h"
#include "fci_mo.h"
#include "dsrg_mrpt2.h"
#include "three_dsrg_mrpt2.h"
#include "tensorsrg.h"
#include "mcsrgpt2_mo.h"
#include "fci_solver.h"
#include "blockedtensorfactory.h"
#include "sq.h"
#include "dsrg_wick.h"
#include "uno.h"

INIT_PLUGIN

void test_davidson();

namespace psi{ namespace libadaptive{

extern "C" int
read_options(std::string name, Options &options)
{
    if (name == "LIBADAPTIVE" || options.read_globals()) {
        /*- MODULEDESCRIPTION Libadaptive */

        /*- SUBSECTION Job Type */

        /// Compute natural orbitals using MP2
        options.add_bool("MP2_NOS",false);
        /// View the natural orbitals with their symmetry information
        options.add_bool("NAT_ORBS_PRINT", false);
        /// Use Natural Orbitals to suggest active space
        options.add_bool("NAT_ACT", false);

        // Natural Orbital selection criteria.  Used to fine tune how many active orbitals there are

        /// Typically, a occupied orbital with a NO occupation of <0.98 is considered active
        options.add_double("OCC_NATURAL", 0.98);
        /// Typically, a virtual orbital with a NO occupation of > 0.02 is considered active
        options.add_double("VIRT_NATURAL", 0.02);

        //////////////////////////////////////////////////////////////
        ///         OPTIONS FOR UNO
        //////////////////////////////////////////////////////////////

        /*- Use unrestricted natural orbitals? -*/
        options.add_bool("UNO", false);
        /*- Minimum occupation number -*/
        options.add_double("UNOMIN", 0.02);
        /*- Maximum occupation number -*/
        options.add_double("UNOMAX", 1.98);
        /*- Print unrestricted natural orbitals -*/
        options.add_bool("UNO_PRINT", false);
        /*- Write Molden -*/
        options.add_bool("MOLDEN_WRITE", false);

        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 0);
        /// Print summary of memory
        options.add_bool("MEMORY_SUMMARY", false);
         

        /*- The algorithm used to screen the determinant
         *  - CONVENTIONAL Conventional two-electron integrals
         *  - DF Density fitted two-electron integrals
         *  - CHOLESKY Cholesky decomposed two-electron integrals -*/
        options.add_str("INT_TYPE","CONVENTIONAL","CONVENTIONAL DF CHOLESKY DISKDF ALL"); 
        
        /* - The tolerance for cholesky integrals */
        options.add_double("CHOLESKY_TOLERANCE", 1e-6);
         
        /*- The job type
         *  - FCI Full configuration interaction (Francesco's code)
         *  - CAS Full configuration interaction (York's code)
         *  - ACI Adaptive configuration interaction
         *  - APICI Adaptive path-integral CI
         *  - DSRG-MRPT2 Tensor-based DSRG-MRPT2 code
        -*/
        options.add_str("JOB_TYPE","EXPLORER","EXPLORER ACI ACI_SPARSE FCIQMC APICI FAPICI FCI CAS"
                                              " SR-DSRG SR-DSRG-ACI SR-DSRG-APICI TENSORSRG TENSORSRG-CI"
                                              " DSRG-MRPT2 MR-DSRG-PT2 THREE-DSRG-MRPT2 SQ NONE");

        /*- The symmetry of the electronic state. (zero based) -*/
        options.add_int("ROOT_SYM",0);

        /*- The multiplicity (2S + 1 )of the electronic state.
         *  For example, 1 = singlet, 2 = doublet, 3 = triplet, ...
         *  If a value is provided it overrides the multiplicity
         *  of the SCF solution. -*/
        options.add_int("MULTIPLICITY",0);

        /*- The charge of the molecule.  If a value is provided
            it overrides the charge of the SCF solution. -*/
        options.add_int("CHARGE",0);

        /*- The minimum excitation level (Default value: 0) -*/
        options.add_int("MIN_EXC_LEVEL",0);

        /*- The maximum excitation level (Default value: 0 = number of electrons) -*/
        options.add_int("MAX_EXC_LEVEL",0);

        /*- Number of frozen occupied orbitals per irrep (in Cotton order) -*/
        options.add("FROZEN_DOCC",new ArrayType());

        /*- Number of restricted doubly occupied orbitals per irrep (in Cotton order) -*/
        options.add("RESTRICTED_DOCC", new ArrayType());

        /*- Number of active orbitals per irrep (in Cotton order) -*/
        options.add("ACTIVE",new ArrayType());

        /*- Number of restricted unoccupied orbitals per irrep (in Cotton order) -*/
        options.add("RESTRICTED_UOCC", new ArrayType());

        /*- Number of frozen unoccupied orbitals per irrep (in Cotton order) -*/
        options.add("FROZEN_UOCC",new ArrayType());

        /*- The algorithm used to screen the determinant
         *  - DENOMINATORS uses the MP denominators to screen strings
         *  - SINGLES generates the space by a series of single excitations -*/
        options.add_str("EXPLORER_ALGORITHM","DENOMINATORS","DENOMINATORS SINGLES");

        /*- The energy threshold for the determinant energy in Hartree -*/
        options.add_double("DET_THRESHOLD",1.0);

        /*- The energy threshold for the MP denominators energy in Hartree -*/
        options.add_double("DEN_THRESHOLD",1.5);

        /*- The criteria used to screen the strings -*/
        options.add_str("SCREENING_TYPE","MP","MP DET");

        // Options for the diagonalization of the Hamiltonian //
        /*- Determines if this job will compute the energy -*/
        options.add_bool("COMPUTE_ENERGY",true);

        /*- The form of the Hamiltonian matrix.
         *  - FIXED diagonalizes a matrix of fixed dimension
         *  - SMOOTH forms a matrix with smoothed matrix elements -*/
        options.add_str("H_TYPE","FIXED_ENERGY","FIXED_ENERGY FIXED_SIZE");

        /*- Determines if this job will compute the energy -*/
        options.add_str("ENERGY_TYPE","FULL","FULL SELECTED LOWDIN SPARSE RENORMALIZE RENORMALIZE_FIXED LMRCISD LMRCIS IMRCISD IMRCISD_SPARSE LMRCISD_SPARSE LMRCIS_SPARSE FACTORIZED_CI");

        /*- The form of the Hamiltonian matrix.
         *  - FIXED diagonalizes a matrix of fixed dimension
         *  - SMOOTH forms a matrix with smoothed matrix elements -*/
        options.add_str("SELECT_TYPE","AMP","ENERGY AMP AIMED_AMP AIMED_ENERGY");

        options.add_int("IMRCISD_TEST_SIZE",0);
        options.add_int("IMRCISD_SIZE",0);

        /*- The number of determinants used to build the Hamiltonian -*/        
        options.add_int("NDETS",100);

        /*- The maximum dimension of the Hamiltonian -*/
        options.add_int("MAX_NDETS",1000000);

        /*- The energy threshold for the model space -*/
        options.add_double("SPACE_M_THRESHOLD",1000.0);

        /*- The energy threshold for the intermdiate space -*/
        options.add_double("SPACE_I_THRESHOLD",1000.0);

        /*- The energy threshold for the intermdiate space -*/
        options.add_double("T2_THRESHOLD",0.000001);

        /*- The threshold for the selection of the P space -*/
        options.add_double("TAUP",0.01);
        /*- The threshold for the selection of the Q space -*/
        options.add_double("TAUQ",0.000001);

        /*- The number of steps used in the renormalized Lambda CI -*/
        options.add_int("RENORMALIZATION_STEPS",10);

        /*- The maximum number of determinant in the fixed-size renormalized Lambda CI -*/
        options.add_int("REN_MAX_NDETS",1000);

        /*- The energy threshold for smoothing the Hamiltonian.
         *  Determinants with energy < DET_THRESHOLD - SMO_THRESHOLD will be included in H
         *  Determinants with DET_THRESHOLD - SMO_THRESHOLD < energy < DET_THRESHOLD will be included in H but smoothed
         *  Determinants with energy > DET_THRESHOLD will not be included in H -*/
        options.add_double("SMO_THRESHOLD",0.0);

        /*- The method used to smooth the Hamiltonian -*/
        options.add_bool("SMOOTH",false);

        /*- The method used to smooth the Hamiltonian -*/
        options.add_bool("SELECT",false);

        /*- The diagonalization method -*/
        options.add_str("DIAG_ALGORITHM","DAVIDSON","DAVIDSON FULL DAVIDSONLIST");

        /*- The number of roots computed -*/
        options.add_int("NROOT",1);

        /*- The root selected for state-specific computations -*/
        options.add_int("ROOT",0);

        // Options for the Cartographer class //
        /*- Density of determinants format -*/
        options.add_str("DOD_FORMAT","HISTOGRAM","GAUSSIAN HISTOGRAM");
        /*- Number of bins used to form the DOD plot -*/
        options.add_int("DOD_BINS",2000);
        /*- Width of the DOD Gaussian/histogram.  Default 0.02 Hartree ~ 0.5 eV -*/
        options.add_double("DOD_BIN_WIDTH",0.05);
        /*- Write an output file? -*/
        options.add_bool("DETTOUR_WRITE_FILE",false);
        /*- Write the determinant occupation? -*/
        options.add_bool("WRITE_OCCUPATION",true);
        /*- Write the determinant energy? -*/
        options.add_bool("WRITE_DET_ENERGY",true);
        /*- Write the denominator energy? -*/
        options.add_bool("WRITE_DEN_ENERGY",false);
        /*- Write the excitation level? -*/
        options.add_bool("WRITE_EXC_LEVEL",false);
        /*- Write information only for a given excitation level.
            0 (default) means print all -*/
        options.add_int("RESTRICT_EXCITATION",0);
        /*- The energy buffer for building the Hamiltonian matrix in Hartree -*/
        options.add_double("H_BUFFER",0.0);

        /*- The maximum number of iterations -*/
        options.add_int("MAXITER",100);

        // Options for the Genetic Algorithm CI //
        /*- The size of the population -*/
        options.add_int("NPOP",100);

        //////////////////////////////////////////////////////////////
        ///         OPTIONS FOR THE FULL CI CODE
        //////////////////////////////////////////////////////////////

        /*- Test the FCI reduced density matrices? -*/
        options.add_bool("TEST_RDMS",false);

        //////////////////////////////////////////////////////////////
        ///         OPTIONS FOR THE ADAPTIVE CI and EX_ACI
        //////////////////////////////////////////////////////////////

        /*- The threshold for smoothing the Hamiltonian. -*/
        options.add_double("SMOOTH_THRESHOLD",0.01);
        /*- The type of selection parameters to use*/
        options.add_bool("PERTURB_SELECT", false);
        /*Function of q-space criteria, per root*/
        options.add_str("PQ_FUNCTION", "AVERAGE","MAX");
        /*Type of  q-space criteria to use (only change for excited states)*/
        options.add_bool("Q_REL", false);
        /*Reference to be used in calculating ∆e (q_rel has to be true)*/
        options.add_str("Q_REFERENCE", "GS", "ADJACENT");
        /* Method to calculate excited state */
        options.add_str("EXCITED_ALGORITHM", "STATE_AVERAGE","ROOT_SELECT SINGLE_STATE");
        /*- Root to determine Q criteria*/
        options.add_int("REF_ROOT",0);
        /*Number of roots to compute on final re-diagonalization*/
        options.add_int("POST_ROOT",1);
        /*Diagonalize after ACI procedure with higher number of roots*/
        options.add_bool("POST_DIAGONALIZE", false);
        /*Maximum number of determinants*/
        options.add_int("MAX_DET", 1e6);
        /*Threshold value for defining multiplicity from S^2*/
        options.add_double("SPIN_TOL", 0.01);
        /*- Compute 1-RDM? -*/
        options.add_bool("1_RDM", false);
        /*- Form initial space with based on energy */
        options.add_bool("LAMBDA_GUESS", false);
		/*- Type of spin projection
		 * 0 - None
		 * 1 - Project initial P spaces at each iteration
		 * 2 - Project only after converged PQ space
		 * 3 - Do 1 and 2 -*/
		options.add_int("SPIN_PROJECTION", 0);
		/*- Threshold for Lambda guess -*/
		options.add_double("LAMBDA_THRESH", 1.0);
		/*- Add determinants to enforce spin-complete set? -*/
		options.add_bool("ENFORCE_SPIN_COMPLETE", false);

        //////////////////////////////////////////////////////////////
        ///         OPTIONS FOR THE ADAPTIVE PATH-INTEGRAL CI
        //////////////////////////////////////////////////////////////
        /*- The propagation algorithm -*/
        options.add_str("PROPAGATOR","LINEAR","LINEAR QUADRATIC CUBIC QUARTIC POWER TROTTER OLSEN DAVIDSON MITRUSHENKOV");
        /*- The determinant importance threshold -*/
        options.add_double("SPAWNING_THRESHOLD",0.001);
        /*- The maximum number of determinants used to form the guess wave function -*/
        options.add_double("MAX_GUESS_SIZE",10000);
        /*- The determinant importance threshold -*/
        options.add_double("GUESS_SPAWNING_THRESHOLD",0.01);
        /*- The threshold with which we estimate the variational energy.
            Note that the final energy is always estimated exactly. -*/
        options.add_double("ENERGY_ESTIMATE_THRESHOLD",1.0e-6);
        /*- The time step in imaginary time (a.u.) -*/
        options.add_double("TAU",0.01);
        /*- The energy convergence criterion -*/
        options.add_double("E_CONVERGENCE",1.0e-8);
        /*- Use a fast (sparse) estimate of the energy -*/
        options.add_bool("FAST_EVAR",false);
        /*- Iterations in between variational estimation of the energy -*/
        options.add_int("ENERGY_ESTIMATE_FREQ",25);
        /*- Use an adaptive time step? -*/
        options.add_bool("ADAPTIVE_BETA",false);
        /*- Use intermediate normalization -*/
        options.add_bool("USE_INTER_NORM",false);
        /*- Use a shift in the exponential -*/
        options.add_bool("USE_SHIFT",false);
        /*- Prescreen the spawning of excitations -*/
        options.add_bool("SIMPLE_PRESCREENING",false);
        /*- Use dynamic prescreening -*/
        options.add_bool("DYNAMIC_PRESCREENING",false);
        /*- The maximum value of beta -*/
        options.add_double("MAXBETA",1000.0);

        //////////////////////////////////////////////////////////////
        ///         OPTIONS FOR THE FULL CI QUANTUM MONTE-CARLO
        //////////////////////////////////////////////////////////////
        /*- The maximum value of beta -*/
        options.add_double("START_NUM_WALKERS",1000.0);
        /*- Spawn excitation type -*/
        options.add_str("SPAWN_TYPE","RANDOM", "RAMDOM ALL GROUND_AND_RANDOM");
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
        options.add_int("VAR_ENERGY_ESTIMATE_FREQ",1000);
        /*- Iterations in between printing information -*/
        options.add_int("PRINT_FREQ",100);

        //////////////////////////////////////////////////////////////
        ///
        ///              OPTIONS FOR THE SRG MODULE
        ///
        //////////////////////////////////////////////////////////////
        /*- The type of operator to use in the SRG transformation -*/
        options.add_str("SRG_MODE","SRG","SRG DSRG CT");
        /*- The type of operator to use in the SRG transformation -*/
        options.add_str("SRG_OP","UNITARY","UNITARY CC");
        /*- The flow generator to use in the SRG equations -*/
        options.add_str("SRG_ETA","WHITE","WEGNER_BLOCK WHITE");
        /*- The integrator used to propagate the SRG equations -*/
        options.add_str("SRG_ODEINT","FEHLBERG78","DOPRI5 CASHKARP FEHLBERG78");
        /*- The end value of the integration parameter s -*/
        options.add_double("SRG_SMAX",10.0);
        /*- The end value of the integration parameter s -*/
        options.add_double("DSRG_S",1.0e10);
        /*- The end value of the integration parameter s -*/
        options.add_double("DSRG_POWER",2.0);


        // --------------------------- SRG EXPERT OPTIONS ---------------------------

        /*- The initial time step used by the ode solver -*/
        options.add_double("SRG_DT",0.001);
        /*- The absolute error tollerance for the ode solver -*/
        options.add_double("SRG_ODEINT_ABSERR",1.0e-12);
        /*- The absolute error tollerance for the ode solver -*/
        options.add_double("SRG_ODEINT_RELERR",1.0e-12);
        /*- Select a modified commutator -*/
        options.add_str("SRG_COMM","STANDARD","STANDARD FO FO2 SRG2");
        /*- The maximum number of commutators in the recursive single commutator approximation -*/
        options.add_int("SRG_RSC_NCOMM",20);
        /*- The treshold for terminating the RSC approximation -*/
        options.add_double("SRG_RSC_THRESHOLD",1.0e-12);
        /*- Save Hbar? -*/
        options.add_bool("SAVE_HBAR",false);



        //////////////////////////////////////////////////////////////
        ///
        ///              OPTIONS FOR THE MR-DSRG-PT2 MODULE
        ///
        //////////////////////////////////////////////////////////////
        /*- Multiplicity -*/
        boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
        int multi = molecule->multiplicity();
        options.add_int("MULTI", multi);            /* multiplicity */
        options.add_int("MS", 0);                   /* Ms value */
        /*- Threshold for Printing CI Vectors -*/
        options.add_double("PRINT_CI_VECTOR", 0.05);
        /*- Semicanonicalize Orbitals -*/
        options.add_bool("SEMI_CANONICAL", true);
        /*- DSRG Taylor Expansion Threshold -*/
        options.add_int("TAYLOR_THRESHOLD", 3);
        /*- DSRG Perturbation -*/
        options.add_bool("DSRGPT", true);
        /*- Print N Largest T Amplitudes -*/
        options.add_int("NTAMP", 15);
        /*- T Threshold for Intruder States -*/
        options.add_double("INTRUDER_TAMP", 0.10);
        /*- Zero T1 Amplitudes -*/
        options.add_bool("T1_ZERO", false);
        /*- The Algorithm to Form T Amplitudes -*/
        options.add_str("T_ALGORITHM", "DSRG", "DSRG DSRG_NOSEMI SELEC ISA");
        /*- Two-Particle Density Cumulant -*/
        options.add_str("TWOPDC", "MK", "MK ZERO");
        /*- Three-Particle Density Cumulant -*/
        options.add_str("THREEPDC", "MK", "MK MK_DECOMP ZERO DIAG");
        /*- Source Operator -*/
        options.add_str("SOURCE", "STANDARD", "STANDARD AMP EMP2 LAMP LEMP2");
        /*- Exponent of Energy Denominator -*/
        options.add_double("DELTA_EXPONENT", 2.0);
        /*- Intruder State Avoidance b Parameter -*/
        options.add_double("ISA_B", 0.02);
        /*- DMRG-CI or CAS-CI reference -*/
        options.add_str("CASTYPE", "CAS", "CAS FCI DMRG");
        /*- Algorithm for the ccvv term for three-dsrg-mrpt2 -*/
        options.add_str("CCVV_ALGORITHM", "FLY_AMBIT", "CORE FLY_AMBIT FLY_LOOP");
        /*- Defintion for source operator for ccvv term -*/
        options.add_str("CCVV_SOURCE", "NORMAL", "ZERO NORMAL");
        /*- Reference Relaxation -*/
        options.add_str("RELAX_REF", "NONE", "NONE ONCE ITERATE");
    }

    return true;
}

extern "C" PsiReturnType
libadaptive(Options &options)
{
    Timer overall_time;
    ambit::initialize();

    if(options.get_bool("UNO")){
        std::string ref = options.get_str("REFERENCE");
        if(ref == "UHF" || ref == "CUHF" || ref == "UKS"){
            UNO uno(options);
        }
    }

    std::shared_ptr<MOSpaceInfo> mo_space_info = std::make_shared<MOSpaceInfo>();
    mo_space_info->read_options(options);

    // Get the one- and two-electron integrals in the MO basis
    // If CHOLESKY
    // create CholeskyIntegrals class
    ExplorerIntegrals* ints_;
    if(options.get_str("INT_TYPE") == "CHOLESKY")

    {
        ints_ = new CholeskyIntegrals(options,UnrestrictedMOs,RemoveFrozenMOs);
    }
    else if(options.get_str("INT_TYPE") == "DF")

    {
        ints_ = new DFIntegrals(options,UnrestrictedMOs,RemoveFrozenMOs);
    }
    else if (options.get_str("INT_TYPE")== "DISKDF")

    {
        ints_ = new DISKDFIntegrals(options,UnrestrictedMOs,RemoveFrozenMOs);
    }

    else 
    {
        ints_ = new ConventionalIntegrals(options,UnrestrictedMOs,RemoveFrozenMOs);
        
    }

    // Link the integrals to the BitsetDeterminant class
    BitsetDeterminant::set_ints(ints_);

    if (options.get_bool("MP2_NOS")){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        MP2_NOS mp2_nos(wfn,options,ints_);
    }

    if (options.get_str("JOB_TYPE") == "MR-DSRG-PT2"){
        MCSRGPT2_MO mcsrgpt2_mo(options, ints_);
    }
    // The explorer object will do its job
    if (options.get_str("JOB_TYPE") == "EXPLORER"){
        LambdaCI* explorer = new LambdaCI(options,ints_);
        delete explorer;
    }
    if (options.get_str("JOB_TYPE") == "FCIQMC"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<FCIQMC> fciqmc(new FCIQMC(wfn,options,ints_));
        fciqmc->compute_energy();
    }
    if ((options.get_str("JOB_TYPE") == "ACI") or (options.get_str("JOB_TYPE") == "ACI_SPARSE")){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<AdaptiveCI> aci(new AdaptiveCI(wfn,options,ints_));
        aci->compute_energy();
    }
    if ((options.get_str("JOB_TYPE") == "EX-ACI")){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<EX_ACI> ex_aci(new EX_ACI(wfn,options,ints_));
        ex_aci->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "APICI"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<AdaptivePathIntegralCI> apici(new AdaptivePathIntegralCI(wfn,options,ints_));
        for (int n = 0; n < options.get_int("NROOT"); ++n){
            apici->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "FAPICI"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<FastAdaptivePathIntegralCI> apici(new FastAdaptivePathIntegralCI(wfn,options,ints_));
        for (int n = 0; n < options.get_int("NROOT"); ++n){
            apici->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "FCI"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<FCI> fci(new FCI(wfn,options,ints_,mo_space_info));
        fci->compute_energy();
    }
    if(options.get_str("JOB_TYPE")=="CAS")
    {
        FCI_MO fci_mo(options,ints_);
    }
    if (options.get_str("JOB_TYPE") == "DSRG-MRPT2"){
        if(options.get_str("CASTYPE")=="CAS")
        {
            FCI_MO fci_mo(options,ints_);
            Reference reference = fci_mo.reference();
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<DSRG_MRPT2> dsrg_mrpt2(new DSRG_MRPT2(reference,wfn,options,ints_));
            dsrg_mrpt2->compute_energy();
            if(options.get_str("RELAX_REF") == "ONCE"){
                boost::shared_ptr<DSRG_WICK> dsrg_wick(new DSRG_WICK(mo_space_info,
                                                                     dsrg_mrpt2->RF(),
                                                                     dsrg_mrpt2->Rtei(),
                                                                     dsrg_mrpt2->Singles(),
                                                                     dsrg_mrpt2->Doubles()));
//                dsrg_mrpt2->transform_integrals();

//                FCI_MO fci(options,ints_);
            }
        }
        if(options.get_str("CASTYPE")=="FCI")
        {
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

            if (options.get_bool("SEMI_CANONICAL")){
                boost::shared_ptr<FCI> fci(new FCI(wfn,options,ints_,mo_space_info));
                fci->compute_energy();
                Reference reference2 = fci->reference();
                SemiCanonical semi(wfn,options,ints_,mo_space_info,reference2);
            }
                boost::shared_ptr<FCI> fci(new FCI(wfn,options,ints_,mo_space_info));
                fci->compute_energy();
                Reference reference = fci->reference();
                boost::shared_ptr<DSRG_MRPT2> dsrg_mrpt2(new DSRG_MRPT2(reference,wfn,options,ints_));
                dsrg_mrpt2->compute_energy();
        }
        else if(options.get_str("CASTYPE")=="DMRG")
        {
            outfile->Printf("\n Buy Kevin a beer and he will maybe implement DMRG into libadaptive\n");
            throw PSIEXCEPTION("DMRG is not available quite yet");
        }

    }
    if (options.get_str("JOB_TYPE") == "THREE_DSRG-MRPT2")
    {

       if(options.get_str("INT_TYPE")=="CONVENTIONAL")
       {
           outfile->Printf("\n THREE_DSRG-MRPT2 is designed for DF/CD integrals");
           throw PSIEXCEPTION("Please set INT_TYPE  DF/CHOLESKY for THREE_DSRG");
       }

       if(options.get_str("CASTYPE")=="CAS")
       {
           FCI_MO fci_mo(options,ints_);
           Reference reference = fci_mo.reference();
           boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
           boost::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(new THREE_DSRG_MRPT2(reference,wfn,options,ints_));
           three_dsrg_mrpt2->compute_energy();
       }

       else if(options.get_str("CASTYPE")=="FCI")
       {
           boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

           if (options.get_bool("SEMI_CANONICAL")){
               boost::shared_ptr<FCI> fci(new FCI(wfn,options,ints_,mo_space_info));
               fci->compute_energy();
               Reference reference2 = fci->reference();
               SemiCanonical semi(wfn,options,ints_,mo_space_info,reference2);
           }
           boost::shared_ptr<FCI> fci(new FCI(wfn,options,ints_,mo_space_info));
           fci->compute_energy();
           Reference reference = fci->reference();
           if(options.get_str("REFERENCE")=="UHF" || options.get_str("REFERENCE")=="CUHF")
           {
                outfile->Printf("\n This method is designed for restricted references (ROHF or RHF)");
                throw PSIEXCEPTION("Use either ROHF or RHF for THREE_DSRG_MRPT2");
           }
           boost::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(new THREE_DSRG_MRPT2(reference,wfn,options,ints_));
           three_dsrg_mrpt2->compute_energy();
       }

       else if(options.get_str("CASTYPE")=="DMRG")

       {
           outfile->Printf("\n Please buy Kevin a beer and maybe he will add DMRG to this code. :-).\n"); 
           throw PSIEXCEPTION("NO DMRG Reference available yet");
       }

    }
    if ((options.get_str("JOB_TYPE") == "TENSORSRG") or (options.get_str("JOB_TYPE") == "SR-DSRG")){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        boost::shared_ptr<TensorSRG> srg(new TensorSRG(wfn,options,ints_));
        srg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "SR-DSRG-ACI"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        {
            auto dsrg = std::make_shared<TensorSRG>(wfn,options,ints_);
            dsrg->compute_energy();
            dsrg->transfer_integrals();
        }
        {
            auto aci = std::make_shared<AdaptiveCI>(wfn,options,ints_);
            aci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "SR-DSRG-APICI"){
        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
        {
            auto dsrg = std::make_shared<TensorSRG>(wfn,options,ints_);
            dsrg->compute_energy();
            dsrg->transfer_integrals();
        }
        {
            auto apici = std::make_shared<AdaptivePathIntegralCI>(wfn,options,ints_);
            apici->compute_energy();
        }
    }

    if (options.get_str("JOB_TYPE") == "SQ"){
        SqTest sqtest;
    }

    // Delete ints_;
    delete ints_;

    ambit::finalize();

    outfile->Printf("\n Your calculation took %8.8f seconds", overall_time.get());
    return Success;
}

}} // End Namespaces
