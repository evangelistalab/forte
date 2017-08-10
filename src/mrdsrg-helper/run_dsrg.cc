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

void set_DSRG_options(ForteOptions& foptions) {

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
     *    - MS:  form 2nd-order Heff_MN = <M|H|N> + 0.5 * [<M|(T_M)^+ H|N> +
     * <M|H T_N|N>]
     *    - XMS: rotate references such that <M|F|N> is diagonal before MS
     * procedure -*/
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
}

/// A uniformed function to run DSRG related jobs
// void run_dsrg() {}
}
}
