#include <cmath>

#include "psi4-dec.h"
#include "psifiles.h"
#include <libplugin/plugin.h>
#include <libdpd/dpd.h>
#include <libpsio/psio.hpp>
#include <libtrans/integraltransform.h>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include "multidimensional_arrays.h"

#include "ga-ci.h"
#include "adaptive-ci.h"
#include "adaptive_pici.h"
#include "lambda-ci.h"
#include "fcimc.h"
#include "sosrg.h"
#include "mosrg.h"
#include "tensorsrg.h"
#include "tensor_test.h"
#include "mcsrgpt2_mo.h"

// This allows us to be lazy in getting the spaces in DPD calls
#define ID(x) ints.DPD_ID(x)

INIT_PLUGIN

void test_davidson();

namespace psi{ namespace libadaptive{

extern "C" int
read_options(std::string name, Options &options)
{
    if (name == "LIBADAPTIVE" || options.read_globals()) {
        /*- MODULEDESCRIPTION Libadaptive */

        /*- SUBSECTION Job Type */
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 0);

        /*- The algorithm used to screen the determinant
         *  - CONVENTIONAL Conventional two-electron integrals
         *  - DF Density fitted two-electron integrals
         *  - CHOLESKY Cholesky decomposed two-electron integrals -*/
        options.add_str("INT_TYPE","CONVENTIONAL","CONVENTIONAL DF CHOLESKY ALL"); 
        
        /* - The tolerance for cholesky integrals */
        options.add_double("CHOLESKY_TOLERANCE", 1e-6);
         
        /*- The job type -*/
        options.add_str("JOB_TYPE","EXPLORER","MR-DSRG-PT2 ACI ACI_SPARSE EXPLORER FCIMC SOSRG SRG SRG-LCI TENSORTEST TENSORSRG TENSORSRG-CI GACI APICI");

        // Options for the Explorer class
        /*- The symmetry of the electronic state. (zero based) -*/
        options.add_int("ROOT_SYM",0);

        /*- The multiplicity of the electronic state.  If a value is provided
            it overrides the multiplicity of the SCF solution. -*/
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

        /*- Number of frozen unoccupied orbitals per irrep (in Cotton order) -*/
        options.add("FROZEN_UOCC",new ArrayType());

        /*- Number of active orbitals per irrep (in Cotton order) -*/
        options.add("ACTIVE",new ArrayType());

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
        options.add_bool("SMOOTH",true);

        /*- The method used to smooth the Hamiltonian -*/
        options.add_bool("SELECT",false);

        /*- The diagonalization method -*/
        options.add_str("DIAG_ALGORITHM","DAVIDSON","DAVIDSON FULL");

        /*- The number of roots computed -*/
        options.add_int("NROOT",4);

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
        ///         OPTIONS FOR THE ADAPTIVE PATH-INTEGRAL CI
        //////////////////////////////////////////////////////////////

        // Options for the Adaptive Path-Integral CI //
        /*- The propagation algorithm -*/
        options.add_str("PROPAGATOR","LINEAR","LINEAR QUADRATIC CUBIC QUARTIC");
        /*- The determinant importance threshold -*/
        options.add_double("SPAWNING_THRESHOLD",0.001);
        /*- The determinant importance threshold -*/
        options.add_double("GUESS_SPAWNING_THRESHOLD",0.01);
        /*- The threshold with which we estimate the variational energy.
            Note that the final energy is always estimated exactly. -*/
        options.add_double("ENERGY_ESTIMATE_THRESHOLD",1.0e-6);
        /*- The time step in imaginary time (a.u.) -*/
        options.add_double("TAU",0.01);
        /*- Use a fast (sparse) estimate of the energy -*/
        options.add_bool("FAST_EVAR",false);
        /*- Iterations in between variational estimation of the energy -*/
        options.add_int("ENERGY_ESTIMATE_FREQ",25);
        /*- Use an adaptive time step? -*/
        options.add_bool("ADAPTIVE_BETA",false);
        /*- Use a shift in the exponential -*/
        options.add_bool("USE_SHIFT",false);
        /*- Prescreen the spawning of excitations -*/
        options.add_bool("SIMPLE_PRESCREENING",false);
        /*- Use dynamic prescreening -*/
        options.add_bool("DYNAMIC_PRESCREENING",false);
        /*- The maximum value of beta -*/
        options.add_double("MAXBETA",1000.0);



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
        options.add_str("THREEPDC", "MK", "MK MK_DECOMP ZERO");
        /*- Intruder State Avoidance b Parameter -*/
        options.add_double("ISA_B", 0.02);
    }

    return true;
}

extern "C" PsiReturnType
libadaptive(Options &options)
{
    if (options.get_str("JOB_TYPE") == "TENSORTEST"){
        test_tensor_class(true);
    }else{
        // Get the one- and two-electron integrals in the MO basis
        ExplorerIntegrals* ints_ = new ExplorerIntegrals(options,UnrestrictedMOs,RemoveFrozenMOs);
        if (options.get_str("JOB_TYPE") == "MR-DSRG-PT2"){
            main::MCSRGPT2_MO mcsrgpt2_mo(options, ints_);
        }
        // The explorer object will do its job
        if (options.get_str("JOB_TYPE") == "EXPLORER"){
            LambdaCI* explorer = new LambdaCI(options,ints_);
            delete explorer;
        }
        if (options.get_str("JOB_TYPE") == "FCIMC"){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<FCIMC> fcimc(new FCIMC(wfn,options,ints_));
            fcimc->compute_energy();
        }
        if ((options.get_str("JOB_TYPE") == "ACI") or (options.get_str("JOB_TYPE") == "ACI_SPARSE")){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<AdaptiveCI> aci(new AdaptiveCI(wfn,options,ints_));
            aci->compute_energy();
        }
        if (options.get_str("JOB_TYPE") == "GACI"){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<GeneticAlgorithmCI> gaci(new GeneticAlgorithmCI(wfn,options,ints_));
            gaci->compute_energy();
        }
        if (options.get_str("JOB_TYPE") == "APICI"){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<AdaptivePathIntegralCI> apici(new AdaptivePathIntegralCI(wfn,options,ints_));
            apici->compute_energy();
        }
        if (options.get_str("JOB_TYPE") == "SOSRG"){
//            Explorer* explorer = new Explorer(options,ints_);
//            std::vector<double> ONa = explorer->Da();
//            std::vector<double> ONb = explorer->Db();
//            int nmo = explorer->nmo();
//            double** G1;
//            init_matrix<double>(G1,2 * nmo,2 * nmo);
//            for (int p = 0; p < nmo; ++p){
//                G1[p][p] = ONa[p];
//                G1[p + nmo][p + nmo] = ONb[p];
//            }
//            SOSRG sosrg(options,ints_,G1);
//            free_matrix<double>(G1,2 * nmo,2 * nmo);

//            delete explorer;
        }
        if (options.get_str("JOB_TYPE") == "TENSORSRG"){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<TensorSRG> srg(new TensorSRG(wfn,options,ints_));
            srg->compute_energy();
        }
        if (options.get_str("JOB_TYPE") == "TENSORSRG-CI"){
            boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
            boost::shared_ptr<TensorSRG> srg(new TensorSRG(wfn,options,ints_));
            srg->compute_energy();
            srg->transfer_integrals();
            LambdaCI* explorer = new LambdaCI(options,ints_);
            delete explorer;
        }
        if (options.get_str("JOB_TYPE") == "SRG"){
            LambdaCI* explorer = new LambdaCI(options,ints_);
            std::vector<double> ONa = explorer->Da();
            std::vector<double> ONb = explorer->Db();
            int ncmo = explorer->ncmo();

            double** G1aa;
            double** G1bb;
            init_matrix<double>(G1aa,ncmo,ncmo);
            init_matrix<double>(G1bb,ncmo,ncmo);
            for (int p = 0; p < ncmo; ++p){
                G1aa[p][p] = ONa[p];
                G1bb[p][p] = ONb[p];
            }
            MOSRG mosrg(options,ints_,G1aa,G1bb);
            free_matrix<double>(G1aa,ncmo,ncmo);
            free_matrix<double>(G1bb,ncmo,ncmo);

            delete explorer;
        }
        if (options.get_str("JOB_TYPE") == "SRG-LCI"){
            double dett = options.get_double("DET_THRESHOLD");
            double dent = options.get_double("DEN_THRESHOLD");
            options.set_double("LIBADAPTIVE","DET_THRESHOLD",1.0e-3);
            options.set_double("LIBADAPTIVE","DEN_THRESHOLD",1.0e-3);

            LambdaCI* explorer = new LambdaCI(options,ints_);
            std::vector<double> ONa = explorer->Da();
            std::vector<double> ONb = explorer->Db();
            int ncmo = explorer->ncmo();

            double** G1aa;
            double** G1bb;
            init_matrix<double>(G1aa,ncmo,ncmo);
            init_matrix<double>(G1bb,ncmo,ncmo);
            for (int p = 0; p < ncmo; ++p){
                G1aa[p][p] = ONa[p];
                G1bb[p][p] = ONb[p];
            }
            MOSRG mosrg(options,ints_,G1aa,G1bb);
            mosrg.transfer_integrals();
            delete explorer;

            options.set_double("LIBADAPTIVE","DET_THRESHOLD",dett);
            options.set_double("LIBADAPTIVE","DEN_THRESHOLD",dent);

            explorer = new LambdaCI(options,ints_);

            free_matrix<double>(G1aa,ncmo,ncmo);
            free_matrix<double>(G1bb,ncmo,ncmo);

            delete explorer;
        }
        // Delete ints_;
        delete ints_;
    }

    return Success;
}

}} // End Namespaces
