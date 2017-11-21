/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "../mrdsrg-so/mrdsrg_so.h"
#include "../mrdsrg-so/so-mrdsrg.h"
#include "../mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "../mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "../mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "../mrdsrg-spin-integrated/mrdsrg.h"
#include "../mrdsrg-spin-integrated/three_dsrg_mrpt2.h"

#include "run_dsrg.h"

namespace psi {
namespace forte {

void set_dsrg_options(ForteOptions& foptions) {

    /*- Correlation level -*/
    foptions.add_str("CORR_LEVEL", "PT2", {"PT2", "PT3", "LDSRG2", "LDSRG2_QC", "LSRG2", "SRG_PT2",
                                           "QDSRG2", "LDSRG2_P3", "QDSRG2_P3"},
                     "Correlation level of MR-DSRG (used in mrdsrg code, "
                     "LDSRG2_P3 and QDSRG2_P3 not implemented)");

    /*- Source Operator -*/
    foptions.add_str("SOURCE", "STANDARD",
                     {"STANDARD", "LABS", "DYSON", "AMP", "EMP2", "LAMP", "LEMP2"},
                     "Source operator used in DSRG (AMP, EMP2, LAMP, LEMP2 "
                     "only available in toy code mcsrgpt2)");

    /*- The Algorithm to Form T Amplitudes -*/
    foptions.add_str("T_ALGORITHM", "DSRG", {"DSRG", "DSRG_NOSEMI", "SELEC", "ISA"},
                     "The way of forming amplitudes (DSRG_NOSEMI, SELEC, ISA "
                     "only available in toy code mcsrgpt2)");

    /*- Different Zeroth-order Hamiltonian -*/
    foptions.add_str("H0TH", "FDIAG", {"FDIAG", "FFULL", "FDIAG_VACTV", "FDIAG_VDIAG"},
                     "Zeroth-order Hamiltonian of DSRG-MRPT (used in mrdsrg code)");

    /*- Compute DSRG dipole momemts -*/
    foptions.add_bool("DSRG_DIPOLE", false, "Compute (if true) DSRG dipole moments");

    /*- Max Iteration for nonperturbative theory -*/
    foptions.add_int("DSRG_MAXITER", 50, "Max iterations for MR-DSRG amplitudes update");

    /*- The residue convergence criterion -*/
    foptions.add_double("R_CONVERGENCE", 1.0e-6, "Convergence criteria for amplitudes");

    /*- Reference Relaxation -*/
    foptions.add_str("RELAX_REF", "NONE", {"NONE", "ONCE", "TWICE", "ITERATE"},
                     "Relax the reference for MR-DSRG (used in dsrg-mrpt2/3, mrdsrg)");

    /*- Max Iteration for Reference Relaxation -*/
    foptions.add_int("MAXITER_RELAX_REF", 15, "Max macro iterations for DSRG reference relaxation");

    /*- DSRG Taylor Expansion Threshold -*/
    foptions.add_int("TAYLOR_THRESHOLD", 3, "Taylor expansion threshold for small denominator");

    /*- Print N Largest T Amplitudes -*/
    foptions.add_int("NTAMP", 15, "Number of amplitudes printed in the summary");

    /*- T Threshold for Intruder States -*/
    foptions.add_double("INTRUDER_TAMP", 0.10,
                        "Threshold for amplitudes considered as intruders for warning");

    /*- DSRG Transformation Type -*/
    foptions.add_str("DSRG_TRANS_TYPE", "UNITARY", {"UNITARY", "CC"}, "DSRG transformation type");

    /*- Automatic Adjusting Flow Parameter -*/
    foptions.add_str("SMART_DSRG_S", "DSRG_S",
                     {"DSRG_S", "MIN_DELTA1", "MAX_DELTA1", "DAVG_MIN_DELTA1", "DAVG_MAX_DELTA1"},
                     "Automatic adjust the flow parameter according to denominators");

    /*- Print DSRG-MRPT3 Timing Profile -*/
    foptions.add_bool("PRINT_TIME_PROFILE", false, "Print detailed timings in dsrg-mrpt3");

    /*- Multi-State DSRG options
     *  - State-average approach
     *    - SA_SUB:  form H_MN = <M|Hbar|N>; M, N are CAS states of interest
     *    - SA_FULL: redo a CASCI
     *  - Multi-state approach (currently only for MRPT2)
     *    - MS:  form 2nd-order Heff_MN = <M|H|N> + 0.5 * [<M|(T_M)^+ H|N> + <M|H T_N|N>]
     *    - XMS: rotate references such that <M|F|N> is diagonal before MS procedure -*/
    foptions.add_str("DSRG_MULTI_STATE", "SA_FULL", {"SA_FULL", "SA_SUB", "MS", "XMS"},
                     "Multi-state DSRG options (MS and XMS recouple states after "
                     "single-state computations)");

    /*- Form 3-Body Hbar (Test for SA_SUB) -*/
    foptions.add_bool("FORM_HBAR3", false,
                      "Form 3-body Hbar (only used in dsrg-mrpt2 with SA_SUB for testing)");

    /*- Form 3-Body Mbar (DSRG dipole) (Test for DSRG-PT2) -*/
    foptions.add_bool("FORM_MBAR3", false,
                      "Form 3-body mbar (only used in dsrg-mrpt2 for testing)");

    /*- DSRG Perturbation -*/
    foptions.add_bool("DSRGPT", true,
                      "Renormalize (if true) the integrals (only used in toy code mcsrgpt2)");

    /*- Include internal amplitudes according to excitation level -*/
    foptions.add_str("INTERNAL_AMP", "NONE", {"NONE", "SINGLES_DOUBLES", "SINGLES", "DOUBLES"},
                     "Include internal amplitudes for VCIS/VCISD-DSRG");

    /*- Select only part of the asked internal amplitudes (IAs) in
     * V-CIS/CISD
     *  - AUTO: all IAs that changes excitations (O->V; OO->VV, OO->OV,
     * OV->VV)
     *  - ALL:  all IAs (O->O, V->V, O->V; OO->OO, OV->OV, VV->VV, OO->VV,
     * OO->OV, OV->VV)
     *  - OOVV: pure external (O->V; OO->VV) -*/
    foptions.add_str("INTERNAL_AMP_SELECT", "AUTO", {"AUTO", "ALL", "OOVV"},
                     "Excitation types considered when internal amplitudes are included");

    /*- T1 Amplitudes -*/
    foptions.add_str("T1_AMP", "DSRG", {"DSRG", "SRG", "ZERO"},
                     "The way of forming T1 amplitudes (used in toy code mcsrgpt2)");

    /*- Intruder State Avoidance b Parameter -*/
    foptions.add_double("ISA_B", 0.02, "Intruder state avoidance parameter "
                                       "when use ISA to form amplitudes (only "
                                       "used in toy code mcsrgpt2)");

    /*- Defintion for source operator for ccvv term -*/
    foptions.add_str("CCVV_SOURCE", "NORMAL", {"ZERO", "NORMAL"},
                     "Special treatment for the CCVV term in DSRG-MRPT2 (used "
                     "in three-dsrg-mrpt2 code)");

    /*- Algorithm for the ccvv term for three-dsrg-mrpt2 -*/
    foptions.add_str("CCVV_ALGORITHM", "FLY_AMBIT",
                     {"CORE", "FLY_AMBIT", "FLY_LOOP", "BATCH_CORE", "BATCH_VIRTUAL",
                      "BATCH_CORE_GA", "BATCH_VIRTUAL_GA", "BATCH_VIRTUAL_MPI", "BATCH_CORE_MPI",
                      "BATCH_CORE_REP", "BATCH_VIRTUAL_REP"},
                     "Algorithm to compute the CCVV term in DSRG-MRPT2 (only "
                     "used in three-dsrg-mrpt2 code)");

    /*- Do AO-DSRG-MRPT2 -*/
    foptions.add_bool("AO_DSRG_MRPT2", false, "Do AO-DSRG-MRPT2 if true (not available)");

    /*- Batches for CCVV_ALGORITHM -*/
    foptions.add_int("CCVV_BATCH_NUMBER", -1, "Batches for CCVV_ALGORITHM");

    /*- Excessive printing for DF_DSRG_MRPT2 -*/
    foptions.add_bool("DSRG_MRPT2_DEBUG", false, "Excssive printing for three-dsrg-mrpt2");

    /*- Algorithm for evaluating 3Cumulant -*/
    foptions.add_str("THREEPDC_ALGORITHM", "CORE", {"CORE", "BATCH"},
                     "Algorithm for evaluating 3-body cumulants in three-dsrg-mrpt2");

    /*- Detailed timing printings -*/
    foptions.add_bool("THREE_MRPT2_TIMINGS", false,
                      "Detailed printing (if true) in three-dsrg-mrpt2");

    /*- Print (1 - exp(-2*s*D)) / D -*/
    foptions.add_bool("PRINT_DENOM2", false,
                      "Print (if true) renormalized denominators in DSRG-MRPT2");

    /*- Do Sequential h_bar evaluation -*/
    foptions.add_bool("DSRG_HBAR_SEQ", false, "Evaluate H_bar sequentially if true");

    /*- Omit blocks with >= 3 virtual indices -*/
    foptions.add_bool("DSRG_OMIT_V3", false, "Omit blocks with >= 3 virtual indices if true");
}

std::shared_ptr<ActiveSpaceSolver>
select_dsrg_actv_solver(SharedWavefunction ref_wfn, Options& options,
                        std::shared_ptr<ForteIntegrals> ints,
                        std::shared_ptr<MOSpaceInfo> mo_space_info) {
    std::shared_ptr<ActiveSpaceSolver> as_solver;
    std::string cas_type = options.get_str("CAS_TYPE");

    if (cas_type == "FCI") {
        as_solver = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
    } else if (cas_type == "CAS") {

    } else if (cas_type == "ACI") {

    } else if (cas_type == "PCI") {

    } else if (cas_type == "V2RDM") {

    } else if (cas_type == "DMRG") {
    }

    return as_solver;
}

std::shared_ptr<DynamicCorrelationSolver>
select_dsrg_code(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
                 std::shared_ptr<MOSpaceInfo> mo_space_info, Reference reference) {
    std::shared_ptr<DynamicCorrelationSolver> dy_solver;
    std::string job_type = options.get_str("JOB_TYPE");

    if (job_type == "DSRG-MRPT2") {
        /// TODO: print a warning if use density fitting.
        /// Once the dipoles are implemented for three-dsrg-pt2, switch to three-dsrg-pt2 if use df.
        dy_solver = std::make_shared<DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
    } else if (job_type == "THREE-DSRG-MRPT2") {
        dy_solver =
            std::make_shared<THREE_DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
    } else if (job_type == "DSRG-MRPT3") {
        dy_solver = std::make_shared<DSRG_MRPT3>(reference, ref_wfn, options, ints, mo_space_info);
    } else if (job_type == "MRDSRG") {
    }

    return dy_solver;
}

void compute_dsrg_energy(SharedWavefunction ref_wfn, Options& options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info) {
    // MK vacuum energy
    auto as_solver = select_dsrg_actv_solver(ref_wfn, options, ints, mo_space_info);
    double Eref = as_solver->compute_energy();

    // MK vacuum density cumulants
    int max_rdm_level = 3;
    if (options.get_str("THREEPDC") == "ZERO") {
        max_rdm_level = 2;
    }

//    /// TODO: need to change all dy_solver to accept Reference&
//    Reference reference = as_solver->reference(max_rdm_level);
    Reference reference;

    /// TODO: add semi-canonicalization here

    // DSRG energy
    auto dy_solver = select_dsrg_code(ref_wfn, options, ints, mo_space_info, reference);

    // DSRG reference relaxation
    int ndsrg = 1, ndiag = 0;
    std::string ref_relax = options.get_str("RELAX_REF");

    /// !!! TODO: need to change keywords for RELAX_REF
    if (ref_relax == "PARTIALLY_RELAXED") {
        ndsrg = 1;
        ndiag = 1;
    } else if (ref_relax == "RELAXED") {
        ndsrg = 2;
        ndiag = 1;
    } else if (ref_relax == "FULLY_RELAXED") {
        ndsrg = options.get_int("MAXITER_RELAX_REF");
        ndiag = ndsrg;
    }

    std::vector<double> Edsrg(ndsrg, 0.0), Ediag(ndiag, 0.0);
    int niter = ndsrg > ndiag ? ndsrg : ndiag;

    for (int i = 0; i < niter; ++i) {
        Edsrg[i] = dy_solver->compute_energy();
        ndsrg -= 1;
        if (ndsrg <= 0 && ndiag <= 0) {
            break;
        }

        auto fci_ints = dy_solver->compute_Heff();

        /// TODO: add semi-canonicalization here

//        /// TODO: pass fci_ints to active-space solver
//        as_solver->set_fci_ints(fci_ints);

        Ediag[i] = as_solver->compute_energy();
        ndiag -= 1;
        if (ndsrg <= 0 && ndiag <= 0) {
            break;
        } else {
//            /// TODO: uncomment reference in ActiveSpaceSolver
//            reference = as_solver->reference(max_rdm_level);

//            /// TODO: may be avoided once dy_solver uses Reference& or shared_ptr<Reference>
//            dy_solver->set_reference(reference);
        }
    }

    // printing
    /// TODO: print reference energy, table of relaxed energies
}
}
}
