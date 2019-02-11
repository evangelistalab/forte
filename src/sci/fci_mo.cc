/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "psi4/libmints/vector.h"
#include "psi4/libmints/dipole.h"
#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/petitelist.h"

#include "sparse_ci/determinant_hashvector.h"
#include "fci/fci_vector.h"
#include "fci_mo.h"
#include "base_classes/forte_options.h"
#include "boost/algorithm/string/predicate.hpp"
#include "sparse_ci/operator.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/iao_builder.h"
#include "helpers/printing.h"
#include "helpers/timer.h"

using namespace psi;

namespace forte {

void set_FCI_MO_options(ForteOptions& foptions) {

    /*- Active space type -*/
    foptions.add_str("FCIMO_ACTV_TYPE", "COMPLETE", {"COMPLETE", "CIS", "CISD", "DOCI"},
                     "The active space type");

    /*- Exclude HF to the CISD space for excited state;
     *  Ground state will be HF energy -*/
    foptions.add_bool("FCIMO_CISD_NOHF", true,
                      "Ground state: HF; Excited states: no HF determinant in CISD space");

    /*- Compute IP/EA in active-CI -*/
    foptions.add_str("FCIMO_IPEA", "NONE", {"NONE", "IP", "EA"}, "Generate IP/EA CIS/CISD space");

    /*- Threshold for printing CI vectors -*/
    foptions.add_double("FCIMO_PRINT_CIVEC", 0.05, "The printing threshold for CI vectors");

    //    /*- Intrinsic atomic orbital analysis -*/
    //    foptions.add_bool("FCIMO_IAO_ANALYSIS", false, "Intrinsic atomic orbital analysis");
}

// FCI_MO::FCI_MO(StateInfo state, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions>
// options,
//               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
//    : ActiveSpaceMethod(state, mo_space_info, as_ints), integral_(ints), scf_info_(scf_info),
//      options_(options) {

//    print_method_banner({"Complete Active Space Configuration Interaction", "Chenyang Li"});
//    startup();

//    // setup integrals
//    fci_ints_ =
//        std::make_shared<ActiveSpaceIntegrals>(integral_,
//        mo_space_info_->get_corr_abs_mo("ACTIVE"),
//                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
//    ambit::Tensor tei_active_aa =
//        integral_->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
//    ambit::Tensor tei_active_ab =
//        integral_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
//    ambit::Tensor tei_active_bb =
//        integral_->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
//    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
//    fci_ints_->compute_restricted_one_body_operator();
//}

FCI_MO::FCI_MO(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), integral_(as_ints->ints()),
      mo_space_info_(mo_space_info), scf_info_(scf_info), options_(options) {

    print_method_banner({"Complete Active Space Configuration Interaction", "Chenyang Li"});
    startup();

    // setup integrals
    if (as_ints != nullptr) {
        fci_ints_ = as_ints;
    } else {
        fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(
            integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
            mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
        ambit::Tensor tei_active_aa =
            integral_->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        ambit::Tensor tei_active_ab =
            integral_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        ambit::Tensor tei_active_bb =
            integral_->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
        fci_ints_->compute_restricted_one_body_operator();
    }
}

FCI_MO::FCI_MO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ActiveSpaceMethod(), integral_(ints), mo_space_info_(mo_space_info), scf_info_(scf_info),
      options_(options) {

    print_method_banner({"Complete Active Space Configuration Interaction", "Chenyang Li"});
    startup();

    // setup integrals
    fci_ints_ =
        std::make_shared<ActiveSpaceIntegrals>(integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    ambit::Tensor tei_active_aa =
        integral_->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    ambit::Tensor tei_active_ab =
        integral_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    ambit::Tensor tei_active_bb =
        integral_->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();
}

FCI_MO::FCI_MO(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
               std::shared_ptr<ActiveSpaceIntegrals> fci_ints)
    : integral_(ints), mo_space_info_(mo_space_info), scf_info_(scf_info), options_(options) {

    print_method_banner({"Complete Active Space Configuration Interaction", "Chenyang Li"});
    startup();

    // setup integrals
    if (fci_ints != nullptr) {
        fci_ints_ = fci_ints;
    } else {
        fci_ints_ = std::make_shared<ActiveSpaceIntegrals>(
            integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
            mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
        ambit::Tensor tei_active_aa =
            integral_->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        ambit::Tensor tei_active_ab =
            integral_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        ambit::Tensor tei_active_bb =
            integral_->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
        fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
        fci_ints_->compute_restricted_one_body_operator();
    }
}

FCI_MO::~FCI_MO() { cleanup(); }

void FCI_MO::cleanup() { clean_all_density_files(); }

void FCI_MO::startup() {
    // read options
    read_options();

    // print options
    if (print_ > 0) {
        print_options();
    }

    // compute orbital extents if CIS/CISD IPEA
    if (ipea_ != "NONE") {
        compute_orbital_extents();
    }
}

void FCI_MO::read_options() {
    // test reference type
    ref_type_ = options_->get_str("REFERENCE");
    if (ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF") {
        outfile->Printf("\n  Unrestricted reference is detected.");
        outfile->Printf("\n  We suggest using unrestricted natural orbitals.");
    }

    // active space type
    actv_space_type_ = options_->get_str("FCIMO_ACTV_TYPE");

    // IP / EA
    ipea_ = options_->get_str("FCIMO_IPEA");

    // print level
    print_ = options_->get_int("PRINT");

    // energy convergence
    econv_ = options_->get_double("E_CONVERGENCE");

    // nuclear repulsion
    e_nuc_ = integral_->nuclear_repulsion_energy();

    // digonalization algorithm
    diag_algorithm_ = options_->get_str("DIAG_ALGORITHM");

    // number of Irrep
    nirrep_ = mo_space_info_->nirrep();
    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    irrep_symbols_.clear();
    for (int h = 0; h < nirrep_; ++h) {
        irrep_symbols_.push_back(ct.gamma(h).symbol());
    }

    // obtain MOs
    nmo_ = mo_space_info_->size("ALL");
    nmopi_ = mo_space_info_->get_dimension("ALL");
    ncmo_ = mo_space_info_->size("CORRELATED");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    // obtain frozen orbitals
    frzc_dim_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    frzv_dim_ = mo_space_info_->get_dimension("FROZEN_UOCC");
    nfrzc_ = mo_space_info_->size("FROZEN_DOCC");
    nfrzv_ = mo_space_info_->size("FROZEN_UOCC");

    // obtain active orbitals
    if ((options_->psi_options())["ACTIVE"].size() == 0) {
        outfile->Printf("\n  Please specify the ACTIVE occupations.");
        outfile->Printf("\n  Single-reference computations should set ACTIVE to zeros.");
        outfile->Printf("\n  For example, ACTIVE [0,0,0,0] depending on the symmetry. \n");
        throw psi::PSIEXCEPTION("Please specify the ACTIVE occupations. Check output for details.");
    }
    actv_dim_ = mo_space_info_->get_dimension("ACTIVE");
    nactv_ = actv_dim_.sum();

    // obitan inactive orbitals
    core_dim_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    virt_dim_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    ncore_ = core_dim_.sum();
    nvirt_ = virt_dim_.sum();

    // compute number of electrons
    std::shared_ptr<psi::Molecule> molecule = psi::Process::environment.molecule();
    int natom = molecule->natom();
    int nelec = 0;
    for (int i = 0; i < natom; ++i) {
        nelec += molecule->fZ(i);
    }
    int charge = molecule->molecular_charge();
    if (options_->has_changed("CHARGE")) {
        charge = options_->get_int("CHARGE");
    }
    nelec -= charge;

    multi_ = molecule->multiplicity();
    if (options_->has_changed("MULTIPLICITY")) {
        multi_ = options_->get_int("MULTIPLICITY");
    }
    if (multi_ < 1) {
        outfile->Printf("\n  MULTIPLICITY must be no less than 1.");
        outfile->Printf("\n  MULTIPLICITY = %2d", multi_);
        outfile->Printf("\n  Check (specify) Multiplicity! \n");
        throw psi::PSIEXCEPTION("MULTIPLICITY must be no less than 1. Check output for details.");
    }
    multi_symbols_ = std::vector<std::string>{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                              "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    twice_ms_ = std::round(2.0 * options_->get_double("MS"));
    if (twice_ms_ < 0) {
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", twice_ms_, multi_);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw psi::PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }

    nalfa_ = (nelec + twice_ms_) / 2;
    nbeta_ = (nelec - twice_ms_) / 2;
    if (nalfa_ < 0 || nbeta_ < 0 || (nalfa_ + nbeta_) != nelec) {
        outfile->Printf("\n  Number of alpha electrons or beta electrons is negative.");
        outfile->Printf("\n  Nalpha = %5ld, Nbeta = %5ld", nalfa_, nbeta_);
        outfile->Printf("\n  Charge = %3d, Multiplicity = %3d, Ms = %.1f", charge, multi_,
                        twice_ms_);
        outfile->Printf("\n  Check the Charge, Multiplicity, and Ms! \n");
        outfile->Printf("\n  Note that Ms is 2 * Sz \n");
        throw psi::PSIEXCEPTION("Negative number of alpha electrons or beta "
                                "electrons. Check output for details.");
    }
    if (nalfa_ - ncore_ - nfrzc_ > nactv_) {
        outfile->Printf("\n  Not enough active orbitals to arrange electrons!");
        outfile->Printf("\n  Number of orbitals: active = %5zu, core = %5zu", nactv_, ncore_);
        outfile->Printf("\n  Number of alpha electrons: Nalpha = %5ld", nalfa_);
        outfile->Printf("\n  Check core and active orbitals! \n");
        throw psi::PSIEXCEPTION("Not enough active orbitals to arrange electrons! "
                                "Check output for details.");
    }

    // obtain root symmetry
    root_sym_ = options_->get_int("ROOT_SYM");

    // obtain number of roots and roots of interest
    //    nroot_ = options_->get_int("NROOT");
    //    root_ = options_->get_int("ROOT");
    if (root_ >= nroot_) {
        outfile->Printf("\n  NROOT = %3d, ROOT = %3d", nroot_, root_);
        outfile->Printf("\n  ROOT must be smaller than NROOT.");
        throw psi::PSIEXCEPTION("ROOT must be smaller than NROOT.");
    }

    // setup symmetry index of active orbitals
    for (int h = 0; h < nirrep_; ++h) {
        for (size_t i = 0; i < size_t(actv_dim_[h]); ++i) {
            sym_actv_.push_back(h);
        }
    }

    // obtain absolute indices of core, active and virtual
    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");

    // active hole and active particle indices
    if (actv_space_type_ == "CIS" || actv_space_type_ == "CISD") {
        psi::Dimension docc_dim(scf_info_->doccpi());
        if (ipea_ == "EA") {
            docc_dim[0] += 1;
        }
        actv_hole_dim_ = docc_dim - frzc_dim_ - core_dim_;
        actv_part_dim_ = actv_dim_ - actv_hole_dim_;

        actv_hole_mos_.clear();
        actv_part_mos_.clear();
        for (int h = 0; h < nirrep_; ++h) {
            int h_local = h;
            size_t offset = 0;
            while (--h_local >= 0) {
                offset += actv_dim_[h_local];
            }

            for (int i = 0; i < actv_dim_[h]; ++i) {
                if (i < actv_hole_dim_[h]) {
                    actv_hole_mos_.push_back(i + offset);
                } else {
                    actv_part_mos_.push_back(i + offset);
                }
            }
        }
    }

    //    // state averaging
    //    if ((options_->psi_options())["AVG_STATE"].size() != 0) {
    //        size_t nstates = 0;
    //        size_t nentry = (options_->psi_options())["AVG_STATE"].size();
    //
    //        // figure out total number of states
    //        std::vector<int> nstatespim;
    //        std::vector<int> irreps;
    //        std::vector<int> multis;
    //        for (size_t i = 0; i < nentry; ++i) {
    //            if ((options_->psi_options())["AVG_STATE"][i].size() != 3) {
    //                outfile->Printf("\n  Error: invalid input of AVG_STATE. Each "
    //                                "entry should take an array of three numbers.");
    //                throw psi::PSIEXCEPTION("Invalid input of AVG_STATE");
    //            }
    //
    //            // irrep
    //            int irrep = (options_->psi_options())["AVG_STATE"][i][0].to_integer();
    //            if (irrep >= nirrep_ || irrep < 0) {
    //                outfile->Printf("\n  Error: invalid irrep in AVG_STATE. Please "
    //                                "check the input irrep (start from 0) not to "
    //                                "exceed %d",
    //                                nirrep_ - 1);
    //                throw psi::PSIEXCEPTION("Invalid irrep in AVG_STATE");
    //            }
    //            irreps.push_back(irrep);
    //
    //            // multiplicity
    //            int multi = (options_->psi_options())["AVG_STATE"][i][1].to_integer();
    //            if (multi < 1) {
    //                outfile->Printf("\n  Error: invalid multiplicity in AVG_STATE.");
    //                throw psi::PSIEXCEPTION("Invaid multiplicity in AVG_STATE");
    //            }
    //            multis.push_back(multi);
    //
    //            // number of states of irrep and multiplicity
    //            int nstates_this = (options_->psi_options())["AVG_STATE"][i][2].to_integer();
    //            if (nstates_this < 1) {
    //                outfile->Printf("\n  Error: invalid nstates in AVG_STATE. "
    //                                "nstates of a certain irrep and multiplicity "
    //                                "should greater than 0.");
    //                throw psi::PSIEXCEPTION("Invalid nstates in AVG_STATE.");
    //            }
    //            nstatespim.push_back(nstates_this);
    //            nstates += nstates_this;
    //        }
    //
    //        // test input weights
    //        std::vector<std::vector<double>> weights;
    //        if ((options_->psi_options())["AVG_WEIGHT"].has_changed()) {
    //            if ((options_->psi_options())["AVG_WEIGHT"].size() != nentry) {
    //                outfile->Printf("\n  Error: mismatched number of entries in "
    //                                "AVG_STATE (%d) and AVG_WEIGHT (%d).",
    //                                nentry, (options_->psi_options())["AVG_WEIGHT"].size());
    //                throw psi::PSIEXCEPTION("Mismatched number of entries in AVG_STATE "
    //                                        "and AVG_WEIGHT.");
    //            }
    //
    //            double wsum = 0.0;
    //            for (size_t i = 0; i < nentry; ++i) {
    //                int nw = (options_->psi_options())["AVG_WEIGHT"][i].size();
    //                if (nw != nstatespim[i]) {
    //                    outfile->Printf("\n  Error: mismatched number of weights "
    //                                    "in entry %d of AVG_WEIGHT. Asked for %d "
    //                                    "states but only %d weights.",
    //                                    i, nstatespim[i], nw);
    //                    throw psi::PSIEXCEPTION("Mismatched number of weights in AVG_WEIGHT.");
    //                }
    //
    //                std::vector<double> weight;
    //                for (int n = 0; n < nw; ++n) {
    //                    double w = (options_->psi_options())["AVG_WEIGHT"][i][n].to_double();
    //                    if (w < 0.0) {
    //                        outfile->Printf("\n  Error: negative weights in AVG_WEIGHT.");
    //                        throw psi::PSIEXCEPTION("Negative weights in AVG_WEIGHT.");
    //                    }
    //                    weight.push_back(w);
    //                    wsum += w;
    //                }
    //                weights.push_back(weight);
    //            }
    //            if (std::fabs(wsum - 1.0) > 1.0e-10) {
    //                outfile->Printf("\n  Error: AVG_WEIGHT entries do not add up "
    //                                "to 1.0. Sum = %.10f",
    //                                wsum);
    //                throw psi::PSIEXCEPTION("AVG_WEIGHT entries do not add up to 1.0.");
    //            }
    //
    //        } else {
    //            // use equal weights
    //            double w = 1.0 / nstates;
    //            for (size_t i = 0; i < nentry; ++i) {
    //                std::vector<double> weight(nstatespim[i], w);
    //                weights.push_back(weight);
    //            }
    //        }
    //
    //        // form option parser
    //        for (size_t i = 0; i < nentry; ++i) {
    //            std::tuple<int, int, int, std::vector<double>> avg_info =
    //                std::make_tuple(irreps[i], multis[i], nstatespim[i], weights[i]);
    //            sa_info_.push_back(avg_info);
    //        }
    //    }
}

void FCI_MO::print_options() {
    print_h2("Input Summary");

    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"No. a electrons in active", nalfa_ - ncore_ - nfrzc_});
    info.push_back({"No. b electrons in active", nbeta_ - ncore_ - nfrzc_});
    info.push_back({"multiplicity", multi_});
    info.push_back({"spin ms (2 * Sz)", twice_ms_});

    for (auto& str_dim : info) {
        outfile->Printf("\n    %-30s = %5zu", str_dim.first.c_str(), str_dim.second);
    }

    print_h2("Orbital Spaces");
    auto print_irrep = [&](const string& str, const psi::Dimension& array) {
        outfile->Printf("\n    %-30s", str.c_str());
        outfile->Printf("[");
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf(" %4d ", array[h]);
        }
        outfile->Printf("]");
    };
    print_irrep("TOTAL MO", nmopi_);
    print_irrep("FROZEN CORE", frzc_dim_);
    print_irrep("FROZEN VIRTUAL", frzv_dim_);
    print_irrep("CORRELATED MO", ncmopi_);
    print_irrep("CORE", core_dim_);
    print_irrep("ACTIVE", actv_dim_);
    print_irrep("VIRTUAL", virt_dim_);

    //    int nentry = sa_info_.size();
    //    if (nentry != 0) {
    //        print_h2("State Averaging Summary");
    //
    //        CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    //        std::vector<std::string> irrep_symbol;
    //        for (int h = 0; h < nirrep_; ++h) {
    //            irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    //        }
    //
    //        int nroots_max = 0;
    //        int nstates = 0;
    //        for (const auto& x : sa_info_) {
    //            int nroots;
    //            std::tie(std::ignore, std::ignore, nroots, std::ignore) = x;
    //            nstates += nroots;
    //            if (nroots > nroots_max) {
    //                nroots_max = nroots;
    //            }
    //        }
    //
    //        if (nroots_max == 1) {
    //            nroots_max = 7;
    //        } else {
    //            nroots_max *= 6;
    //            nroots_max -= 1;
    //        }
    //        int ltotal = 6 + 2 + 6 + 2 + 7 + 2 + nroots_max;
    //        std::string blank(nroots_max - 7, ' ');
    //        std::string dash(ltotal, '-');
    //        outfile->Printf("\n    Irrep.  Multi.  Nstates  %sWeights", blank.c_str());
    //        outfile->Printf("\n    %s", dash.c_str());
    //
    //        for (int i = 0; i < nentry; ++i) {
    //            int irrep, multi, nroots;
    //            std::vector<double> weights;
    //            std::tie(irrep, multi, nroots, weights) = sa_info_[i];
    //
    //            std::string w_str;
    //            for (const double& w : weights) {
    //                std::stringstream ss;
    //                ss << std::fixed << std::setprecision(3) << w;
    //                w_str += ss.str() + " ";
    //            }
    //            w_str.pop_back(); // delete the last space character
    //
    //            std::stringstream ss;
    //            ss << std::setw(4) << std::right << irrep_symbol[irrep] << "    " << std::setw(4)
    //               << std::right << multi << "    " << std::setw(5) << std::right << nroots << " "
    //               << std::setw(nroots_max) << w_str;
    //            outfile->Printf("\n    %s", ss.str().c_str());
    //        }
    //        outfile->Printf("\n    %s", dash.c_str());
    //        outfile->Printf("\n    Total number of states: %d", nstates);
    //        outfile->Printf("\n    %s\n", dash.c_str());
    //    }
}

double FCI_MO::compute_energy() {

    energies_ = compute_ss_energies();
    psi::Process::environment.globals["CURRENT ENERGY"] = Eref_;
    psi::Process::environment.globals["FCI_MO ENERGY"] = Eref_;

    //   energies_.resize(nroot_,0.0);
    //   for( int n = 0; n < nroot_; ++n ){
    //       energies_[n] = eigen_[n].second;
    //   }
    Eref_ = energies_[root_];
    return Eref_;
}

std::vector<double> FCI_MO::compute_ss_energies() {
    // form determinants
    form_p_space();

    // diagonalize the CASCI Hamiltonian
    bool noHF = options_->get_bool("FCIMO_CISD_NOHF");
    if (multi_ == 1 && root_sym_ == 0 &&
        (actv_space_type_ == "CIS" || (actv_space_type_ == "CISD" && noHF))) {
        Diagonalize_H_noHF(determinant_, multi_, nroot_, eigen_);
    } else {
        Diagonalize_H(determinant_, multi_, nroot_, eigen_);
    }

    // print CI vectors in eigen_
    size_t eigen_size = eigen_.size();
    if (static_cast<size_t>(nroot_) > eigen_size) {
        outfile->Printf("\n  Too many roots of interest!");
        std::string be = (eigen_size > 1) ? "are" : "is";
        std::string plural = (eigen_size > 1) ? "roots" : "root";
        outfile->Printf("\n  There %s only %3d %s that satisfy the condition!", be.c_str(),
                        eigen_size, plural.c_str());
        outfile->Printf("\n  Check root_sym, multi, and number of determinants.");
        throw psi::PSIEXCEPTION("Too many roots of interest.");
    }
    print_CI(nroot_, options_->get_double("FCIMO_PRINT_CIVEC"), eigen_, determinant_);

    // compute dipole moments
    compute_permanent_dipole();

    // compute oscillator strength
    if (nroot_ > 1) {
        compute_transition_dipole();
        compute_oscillator_strength();
    }

    double Eref = eigen_[root_].second;
    Eref_ = Eref;
    psi::Process::environment.globals["CURRENT ENERGY"] = Eref;

    // Return just the energies
    std::vector<double> en;
    for (auto& p : eigen_) {
        en.push_back(p.second);
    }
    return en;
}

void FCI_MO::form_p_space() {
    // clean previous determinants
    determinant_.clear();

    // form determinants
    if (actv_space_type_ == "CIS") {
        form_det_cis();
    } else if (actv_space_type_ == "CISD") {
        form_det_cisd();
    } else {
        form_det();
    }

    // printing
    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"number of alpha active electrons", nalfa_ - ncore_ - nfrzc_});
    info.push_back({"number of beta active electrons", nbeta_ - ncore_ - nfrzc_});
    info.push_back({"root symmetry (zero based)", root_sym_});
    if (root_sym_ == 0 && (actv_space_type_ == "CIS" || actv_space_type_ == "CISD")) {
        info.push_back({"number of determinants (include RHF)", determinant_.size()});
    } else {
        info.push_back({"number of determinants", determinant_.size()});
    }

    if (!quiet_) {
        print_h2("Determinants Summary");
        for (auto& str_dim : info) {
            outfile->Printf("\n    %-40s = %5zu", str_dim.first.c_str(), str_dim.second);
        }
        outfile->Printf("\n");
    }

    if (print_ > 1) {
        print_det(determinant_);
    }

    if (determinant_.size() == 0) {
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Check the wavefunction symmetry and multiplicity.");
        throw psi::PSIEXCEPTION("No determinant matching the conditions!");
    }
}

void FCI_MO::form_det() {
    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - ncore_ - nfrzc_;
    int nb_a = nbeta_ - ncore_ - nfrzc_;

    // Alpha and Beta Strings
    local_timer tstrings;
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", "Forming alpha and beta strings");
    }
    std::vector<vector<vector<bool>>> a_string = Form_String(na_a);
    std::vector<vector<vector<bool>>> b_string = Form_String(nb_a);
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tstrings.get());
    }

    // Form Determinant
    local_timer tdet;
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", "Forming determinants");
    }
    if (actv_space_type_ == "DOCI") {
        if (root_sym_ != 0 || multi_ != 1) {
            outfile->Printf("\n  State must be totally symmetric for DOCI.");
            throw psi::PSIEXCEPTION("State must be totally symmetric for DOCI.");
        } else {
            for (int i = 0; i != nirrep_; ++i) {
                size_t sa = a_string[i].size();
                for (size_t alfa = 0; alfa < sa; ++alfa) {
                    determinant_.push_back(Determinant(a_string[i][alfa], a_string[i][alfa]));
                }
            }
        }
    } else {
        for (int i = 0; i != nirrep_; ++i) {
            int j = i ^ root_sym_;
            size_t sa = a_string[i].size();
            size_t sb = b_string[j].size();
            for (size_t alfa = 0; alfa < sa; ++alfa) {
                for (size_t beta = 0; beta < sb; ++beta) {
                    determinant_.push_back(Determinant(a_string[i][alfa], b_string[j][beta]));
                }
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }
}

vector<vector<vector<bool>>> FCI_MO::Form_String(const int& active_elec, const bool& print) {
    timer_on("FORM String");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // initalize the string (only active)
    int symmetry = 0;
    bool* I_init = new bool[nactv_];
    for (size_t i = 0; i < nactv_; ++i)
        I_init[i] = 0;
    for (size_t i = nactv_ - active_elec; i < nactv_; ++i)
        I_init[i] = 1;

    do {
        // permute the active
        std::vector<bool> string_a;
        int sym = symmetry;
        for (size_t i = 0; i < nactv_; ++i) {
            string_a.push_back(I_init[i]);
            if (I_init[i] == 1) {
                sym ^= sym_actv_[i];
            }
        }
        String[sym].push_back(string_a);
    } while (std::next_permutation(I_init, I_init + nactv_));

    if (print == true && !quiet_) {
        print_occupation_strings_perirrep("Possible Strings", String);
    }

    delete[] I_init;
    timer_off("FORM String");
    return String;
}

void FCI_MO::form_det_cis() {
    // reference string
    std::vector<bool> string_ref = Form_String_Ref();

    // singles string
    std::vector<vector<vector<bool>>> string_singles;
    if (ipea_ == "IP") {
        string_singles = Form_String_IP(string_ref);
    } else if (ipea_ == "EA") {
        string_singles = Form_String_EA(string_ref);
    } else {
        string_singles = Form_String_Singles(string_ref);
    }

    // symmetry of ref (just active)
    int symmetry = 0;
    for (size_t i = 0; i < nactv_; ++i) {
        if (string_ref[i]) {
            symmetry ^= sym_actv_[i];
        }
    }

    // singles
    local_timer tdet;
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", "Forming determinants");
    }

    int i = symmetry ^ root_sym_;
    size_t single_size = string_singles[i].size();
    for (size_t x = 0; x < single_size; ++x) {
        determinant_.push_back(Determinant(string_singles[i][x], string_ref));
        determinant_.push_back(Determinant(string_ref, string_singles[i][x]));
    }

    // add HF determinant at the end if root_sym = 0
    if (root_sym_ == 0) {
        determinant_.push_back(Determinant(string_ref, string_ref));
    }

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }
}

void FCI_MO::form_det_cisd() {
    // reference string
    std::vector<bool> string_ref = Form_String_Ref();

    // singles string
    std::vector<vector<vector<bool>>> string_singles = Form_String_Singles(string_ref);
    std::vector<vector<vector<bool>>> string_singles_ipea;
    if (ipea_ == "IP") {
        string_singles_ipea = Form_String_IP(string_ref);
    } else if (ipea_ == "EA") {
        string_singles_ipea = Form_String_EA(string_ref);
    }

    // doubles string
    std::vector<vector<vector<bool>>> string_doubles = Form_String_Doubles(string_ref);

    // symmetry of ref (just active)
    int symmetry = 0;
    for (size_t i = 0; i < nactv_; ++i) {
        if (string_ref[i]) {
            symmetry ^= sym_actv_[i];
        }
    }

    // singles
    local_timer tdet;
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", "Forming determinants");
    }

    int i = symmetry ^ root_sym_;
    singles_size_ = 0;
    if (ipea_ == "NONE") {
        size_t single_size = string_singles[i].size();
        singles_size_ = 2 * single_size;
        for (size_t x = 0; x < single_size; ++x) {
            determinant_.push_back(Determinant(string_singles[i][x], string_ref));
            determinant_.push_back(Determinant(string_ref, string_singles[i][x]));
        }
    } else {
        size_t single_size = string_singles_ipea[i].size();
        singles_size_ = 2 * single_size;
        for (size_t x = 0; x < single_size; ++x) {
            determinant_.push_back(Determinant(string_singles_ipea[i][x], string_ref));
            determinant_.push_back(Determinant(string_ref, string_singles_ipea[i][x]));
        }
    }

    // doubles
    size_t double_size = string_doubles[i].size();
    for (size_t x = 0; x < double_size; ++x) {
        determinant_.push_back(Determinant(string_doubles[i][x], string_ref));
        determinant_.push_back(Determinant(string_ref, string_doubles[i][x]));
    }

    for (int h = 0; h < nirrep_; ++h) {
        size_t single_size_a = string_singles[h].size();
        for (size_t x = 0; x < single_size_a; ++x) {
            int sym = h ^ root_sym_;

            size_t single_size_b = string_singles[sym].size();
            for (size_t y = 0; y < single_size_b; ++y) {
                determinant_.push_back(Determinant(string_singles[h][x], string_singles[sym][y]));
            }

            if (ipea_ != "NONE") {
                size_t single_ipea_size_b = string_singles_ipea[sym].size();
                for (size_t y = 0; y < single_ipea_size_b; ++y) {
                    determinant_.push_back(
                        Determinant(string_singles[h][x], string_singles_ipea[sym][y]));
                    determinant_.push_back(
                        Determinant(string_singles_ipea[sym][y], string_singles[h][x]));
                }
            }
        }
    }

    // add HF determinant at the end if root_sym = 0
    if (root_sym_ == 0) {
        determinant_.push_back(Determinant(string_ref, string_ref));
    }

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }
}

vector<bool> FCI_MO::Form_String_Ref(const bool& print) {
    timer_on("FORM String Ref");

    std::vector<bool> String;
    for (int h = 0; h < nirrep_; ++h) {
        int act_docc = actv_hole_dim_[h];
        int act = actv_dim_[h];
        for (int i = 0; i < act; ++i) {
            String.push_back(i < act_docc);
        }
    }

    if (print) {
        print_h2("Reference String");
        outfile->Printf("    ");
        for (bool b : String) {
            outfile->Printf("%d ", b);
        }
    }

    timer_off("FORM String Ref");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_Singles(const std::vector<bool>& ref_string,
                                                         const bool& print) {
    timer_on("FORM String Singles");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    std::vector<int> uocc, occ;
    for (size_t i = 0; i < nactv_; ++i) {
        if (ipea_ != "NONE" &&
            std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()) {
            continue;
        }

        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_actv_[i];
        } else {
            uocc.push_back(i);
        }
    }

    // singles
    for (const int& a : uocc) {
        std::vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_actv_[a];
        for (const int& i : occ) {
            string_local[i] = false;
            sym ^= sym_actv_[i];
            String[sym].push_back(string_local);
            // need to reset
            string_local[i] = true;
            sym ^= sym_actv_[i];
        }
    }

    if (print) {
        print_occupation_strings_perirrep("Singles Strings", String);
    }

    timer_off("FORM String Singles");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_IP(const std::vector<bool>& ref_string,
                                                    const bool& print) {
    timer_on("FORM String Singles IP");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    std::vector<int> occ;
    for (size_t i = 0; i < nactv_; ++i) {
        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_actv_[i];
        }
    }

    // singles
    for (const int& i : occ) {
        std::vector<bool> string_local(ref_string);
        string_local[idx_diffused_] = true;

        string_local[i] = false;
        int sym = symmetry ^ sym_actv_[i];
        String[sym].push_back(string_local);
    }

    if (print) {
        print_occupation_strings_perirrep("Singles Strings IP", String);
    }

    timer_off("FORM String Singles IP");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_EA(const std::vector<bool>& ref_string,
                                                    const bool& print) {
    timer_on("FORM String Singles EA");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    std::vector<int> uocc;
    for (size_t i = 0; i < nactv_; ++i) {
        if (!ref_string[i]) {
            uocc.push_back(i);
        } else {
            symmetry ^= sym_actv_[i];
        }
    }

    // singles
    for (const int& a : uocc) {
        std::vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_actv_[a];

        string_local[idx_diffused_] = false;
        String[sym].push_back(string_local);
    }

    if (print) {
        print_occupation_strings_perirrep("Singles Strings EA", String);
    }

    timer_off("FORM String Singles EA");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_Doubles(const std::vector<bool>& ref_string,
                                                         const bool& print) {
    timer_on("FORM String Doubles");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    std::vector<int> uocc, occ;
    for (size_t i = 0; i < nactv_; ++i) {
        if (ipea_ != "NONE" && i != idx_diffused_ &&
            std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()) {
            continue;
        }

        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_actv_[i];
        } else {
            uocc.push_back(i);
        }
    }

    // doubles
    for (const int& a : uocc) {
        std::vector<bool> string_a(ref_string);
        string_a[a] = true;
        int sym_a = symmetry ^ sym_actv_[a];

        for (const int& b : uocc) {
            if (b > a) {
                std::vector<bool> string_b(string_a);
                string_b[b] = true;
                int sym_b = sym_a ^ sym_actv_[b];

                for (const int& i : occ) {
                    std::vector<bool> string_i(string_b);
                    string_i[i] = false;
                    int sym_i = sym_b ^ sym_actv_[i];

                    for (const int& j : occ) {
                        if (j > i) {
                            std::vector<bool> string_j(string_i);
                            string_j[j] = false;
                            int sym_j = sym_i ^ sym_actv_[j];
                            String[sym_j].push_back(string_j);
                        }
                    }
                }
            }
        }
    }

    if (print) {
        print_occupation_strings_perirrep("Doubles Strings", String);
    }

    timer_off("FORM String Doubles");
    return String;
}

vector<double> FCI_MO::compute_T1_percentage() {
    std::vector<double> out;

    if (actv_space_type_ != "CISD") {
        outfile->Printf("\n  No point to compute T1 percentage. Return an empty vector.");
    } else {
        // in consistent to form_det_cisd,
        // the first singles_size_ determinants in determinant_ are singles
        for (size_t n = 0, eigen_size = eigen_.size(); n < eigen_size; ++n) {
            double t1 = 0;
            psi::SharedVector evec = eigen_[n].first;
            for (size_t i = 0; i < singles_size_; ++i) {
                double v = evec->get(i);
                t1 += v * v;
            }
            out.push_back(100.0 * t1);
        }
    }

    return out;
}

void FCI_MO::Diagonalize_H_noHF(const vecdet& p_space, const int& multi, const int& nroot,
                                std::vector<pair<psi::SharedVector, double>>& eigen) {
    // recompute RHF determinant
    std::vector<bool> string_ref = Form_String_Ref();
    Determinant rhf(string_ref, string_ref);

    // test if RHF determinant is the last one in det
    Determinant det_back(p_space.back());
    if (rhf == det_back) {
        eigen.clear();
        size_t det_size = p_space.size();

        // compute RHF energy
        outfile->Printf("\n  Isolate RHF determinant to the rest determinants.");
        outfile->Printf("\n  Recompute RHF energy ... ");
        double Erhf = fci_ints_->energy(rhf) + fci_ints_->scalar_energy() + e_nuc_;
        psi::SharedVector rhf_vec(new psi::Vector("RHF Eigen Vector", det_size));
        rhf_vec->set(det_size - 1, 1.0);
        eigen.push_back(std::make_pair(rhf_vec, Erhf));
        outfile->Printf("Done.");

        // compute the rest of the states
        if (nroot > 1) {
            outfile->Printf("\n  The upcoming diagonalization excludes RHF determinant.\n");

            int nroot_noHF = nroot - 1;
            vecdet p_space_noHF(p_space);
            p_space_noHF.pop_back();
            std::vector<pair<psi::SharedVector, double>> eigen_noHF;
            Diagonalize_H(p_space_noHF, multi, nroot_noHF, eigen_noHF);

            for (int i = 0; i < nroot_noHF; ++i) {
                psi::SharedVector vec_noHF = eigen_noHF[i].first;
                double Ethis = eigen_noHF[i].second;

                string name = "Root " + std::to_string(i) + " Eigen Vector";
                psi::SharedVector vec(new psi::Vector(name, det_size));
                for (size_t n = 0; n < det_size - 1; ++n) {
                    vec->set(n, vec_noHF->get(n));
                }

                eigen.push_back(std::make_pair(vec, Ethis));
            }
        }

    } else {
        outfile->Printf("\n  Error: RHF determinant NOT at the end of the determinant vector.");
        outfile->Printf("\n    Diagonalize_H_noHF only works for root_sym = 0.");
        throw psi::PSIEXCEPTION("RHF determinant NOT at the end of determinant vector. "
                                "Problem at Diagonalize_H_noHF of FCI_MO.");
    }
}

void FCI_MO::Diagonalize_H(const vecdet& p_space, const int& multi, const int& nroot,
                           std::vector<pair<psi::SharedVector, double>>& eigen) {
    timer_on("Diagonalize H");
    local_timer tdiagH;
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", "Diagonalizing Hamiltonian");
    }
    size_t det_size = p_space.size();
    eigen.clear();

    // DL solver
    SparseCISolver sparse_solver(fci_ints_);
    DiagonalizationMethod diag_method = DLSolver;
    string sigma_method = options_->get_str("SIGMA_BUILD_TYPE");
    sparse_solver.set_e_convergence(econv_);
    sparse_solver.set_spin_project(true);
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver.set_sigma_method(sigma_method);
    if (projected_roots_.size() != 0) {
        sparse_solver.set_root_project(true);
        sparse_solver.add_bad_states(projected_roots_);
    }
    if (initial_guess_.size() != 0) {
        if (initial_guess_.size() == p_space.size()) {
            sparse_solver.set_initial_guess(initial_guess_);
        }
    }
    if (!quiet_) {
        sparse_solver.set_print_details(true);
    }

    // setup eigen values and vectors
    psi::SharedMatrix evecs;
    psi::SharedVector evals;

    // diagnoalize the Hamiltonian
    if (det_size <= 200) {
        // full Hamiltonian if detsize <= 200
        diag_method = Full;
        sparse_solver.diagonalize_hamiltonian(p_space, evals, evecs, nroot, multi, diag_method);
    } else {
        // use determinant map
        DeterminantHashVec detmap(p_space);
        auto act_mo = mo_space_info_->symmetry("ACTIVE");
        WFNOperator op(act_mo, fci_ints_);
        op.build_strings(detmap);
        if (sigma_method == "HZ") {
            op.op_lists(detmap);
            op.tp_lists(detmap);
        } else {
            op.op_s_lists(detmap);
            op.tp_s_lists(detmap);
        }

        sparse_solver.diagonalize_hamiltonian_map(detmap, op, evals, evecs, nroot, multi,
                                                  diag_method);
    }

    // fill in eigen (spin is purified in DL solver)
    double energy_offset = fci_ints_->scalar_energy() + e_nuc_;
    for (int i = 0; i != nroot; ++i) {
        double value = evals->get(i);
        eigen.push_back(std::make_pair(evecs->get_column(0, i), value + energy_offset));
    }

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdiagH.get());
    }
    timer_off("Diagonalize H");
}

void FCI_MO::print_CI(const int& nroot, const double& CI_threshold,
                      const std::vector<pair<psi::SharedVector, double>>& eigen,
                      const vecdet& det) {
    timer_on("Print CI Vectors");
    if (!quiet_) {
        outfile->Printf("\n\n  * * * * * * * * * * * * * * * * *");
        outfile->Printf("\n  *  CI Vectors & Configurations  *");
        outfile->Printf("\n  * * * * * * * * * * * * * * * * *");
        outfile->Printf("\n");
    }

    dominant_dets_.clear();
    for (int i = 0; i != nroot; ++i) {
        std::vector<std::tuple<double, int>> ci_select; // tuple<coeff, index>

        // choose CI coefficients greater than CI_threshold
        for (size_t j = 0, det_size = det.size(); j < det_size; ++j) {
            double value = (eigen[i].first)->get(j);
            if (std::fabs(value) > CI_threshold)
                ci_select.push_back(std::make_tuple(value, j));
        }
        std::sort(ci_select.begin(), ci_select.end(),
                  [](const std::tuple<double, int>& lhs, const std::tuple<double, int>& rhs) {
                      return std::fabs(std::get<0>(lhs)) > std::fabs(std::get<0>(rhs));
                  });
        dominant_dets_.push_back(det[std::get<1>(ci_select[0])]);

        if (!quiet_) {
            outfile->Printf("\n  ==> Root No. %d <==\n", i);
            for (size_t j = 0, ci_select_size = ci_select.size(); j < ci_select_size; ++j) {
                outfile->Printf("\n    ");
                double ci = std::get<0>(ci_select[j]);
                size_t index = std::get<1>(ci_select[j]);
                size_t ncmopi = 0;
                for (int h = 0; h < nirrep_; ++h) {
                    for (int k = 0; k < actv_dim_[h]; ++k) {
                        size_t x = k + ncmopi;
                        bool a = det[index].get_alfa_bit(x);
                        bool b = det[index].get_beta_bit(x);
                        if (a == b) {
                            outfile->Printf("%d", a == 1 ? 2 : 0);
                        } else {
                            outfile->Printf("%c", a == 1 ? 'a' : 'b');
                        }
                    }
                    if (actv_dim_[h] != 0)
                        outfile->Printf(" ");
                    ncmopi += actv_dim_[h];
                }
                outfile->Printf(" %20.10f", ci);
            }
            outfile->Printf("\n\n    Total Energy:   %.15lf\n\n", eigen[i].second);
        }
    }

    timer_off("Print CI Vectors");
}

void FCI_MO::compute_permanent_dipole() {

    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::string irrep_symbol = ct.gamma(root_sym_).symbol();
    std::string title = "Permanent Dipole Moments (" + irrep_symbol + ")";
    print_h2(title);
    outfile->Printf("\n  Only print nonzero (> 1.0e-5) elements.");

    // obtain AO dipole from ForteIntegrals
    std::vector<psi::SharedMatrix> aodipole_ints = integral_->AOdipole_ints();

    // Nuclear dipole contribution
    Vector3 ndip =
        psi::Process::environment.molecule()->nuclear_dipole(psi::Vector3(0.0, 0.0, 0.0));
    //        DipoleInt::nuclear_contribution(psi::Process::environment.molecule(), );

    // SO to AO transformer
    psi::SharedMatrix sotoao(integral_->aotoso()->transpose());

    // prepare eigen vectors for ci_rdm
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
    for (size_t i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }

    // loop over states
    for (int A = 0; A < nroot_; ++A) {
        std::string trans_name = std::to_string(A) + " -> " + std::to_string(A);

        CI_RDMS ci_rdms(fci_ints_, determinant_, evecs, A, A);
        std::vector<double> opdm_a, opdm_b;
        ci_rdms.compute_1rdm(opdm_a, opdm_b);

        psi::SharedMatrix SOdens = reformat_1rdm("SO density " + trans_name, opdm_a, false);
        SOdens->back_transform(integral_->Ca());

        size_t nao = sotoao->coldim(0);
        psi::SharedMatrix AOdens(new psi::Matrix("AO density " + trans_name, nao, nao));
        AOdens->remove_symmetry(SOdens, sotoao);

        std::vector<double> de(4, 0.0);
        for (int i = 0; i < 3; ++i) {
            de[i] = 2.0 * AOdens->vector_dot(aodipole_ints[i]); // 2.0 for beta spin
            de[i] += ndip[i];                                   // add nuclear contributions
            de[3] += de[i] * de[i];                             // store de * de in the fourth dim
        }
        de[3] = sqrt(de[3]);

        if (de[3] > 1.0e-5) {
            outfile->Printf("\n  Permanent dipole moments (a.u.) %s:  X: %7.4f  Y: "
                            "%7.4f  Z: %7.4f  Total: %7.4f",
                            trans_name.c_str(), de[0], de[1], de[2], de[3]);
        }
    }
    outfile->Printf("\n");
}

psi::SharedMatrix FCI_MO::reformat_1rdm(const std::string& name, const std::vector<double>& data,
                                        bool TrD) {
    psi::SharedMatrix rdm(new psi::Matrix(name, nmopi_, nmopi_));

    // active
    size_t offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        size_t offset1 = frzc_dim_[h] + core_dim_[h];
        for (int u = 0; u < actv_dim_[h]; ++u) {
            size_t mu = u + offset;
            size_t nu = u + offset1;

            for (int v = 0; v < actv_dim_[h]; ++v) {
                size_t mv = v + offset;
                size_t nv = v + offset1;

                rdm->set(h, nu, nv, data[mu * nactv_ + mv]);
            }
        }
        offset += actv_dim_[h];
    }

    if (!TrD) {
        for (int h = 0; h < nirrep_; ++h) {
            // frozen core
            for (int i = 0; i < frzc_dim_[h]; ++i) {
                rdm->set(h, i, i, 1.0);
            }

            // restricted core
            size_t offset1 = frzc_dim_[h];
            for (int i = 0; i < core_dim_[h]; ++i) {
                size_t ni = i + offset1;
                rdm->set(h, ni, ni, 1.0);
            }
        }
    }

    return rdm;
}

void FCI_MO::compute_transition_dipole() {
    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::string irrep_symbol = ct.gamma(root_sym_).symbol();
    std::stringstream title;
    title << "Transition Dipole Moments (" << irrep_symbol << " -> " << irrep_symbol << ")";
    print_h2(title.str());
    outfile->Printf("\n  Only print nonzero (> 1.0e-5) elements.");

    // obtain AO dipole from libmints
    std::vector<psi::SharedMatrix> aodipole_ints = integral_->AOdipole_ints();

    // SO to AO transformer
    psi::SharedMatrix sotoao(integral_->aotoso()->transpose());

    //    // obtain SO dipole from libmints
    //    std::vector<psi::SharedMatrix> dipole_ints;
    //    for(const std::string& direction: {"X","Y","Z"}){
    //        std::string name = "SO Dipole" + direction;
    //        dipole_ints.push_back(std::make_shared<psi::Matrix>(name, this->nsopi(),
    //        this->nsopi()) ));
    //    }

    //    std::shared_ptr<psi::BasisSet> basisset = this->basisset();
    //    std::shared_ptr<IntegralFactory> ints =
    //    std::shared_ptr<IntegralFactory>(
    //                new IntegralFactory(basisset,basisset,basisset,basisset));
    //    std::shared_ptr<OneBodySOInt> sodOBI(ints->so_dipole());

    //    Vector3 origin (0.0, 0.0, 0.0);
    //    sodOBI->ob()->set_origin(origin);
    //    sodOBI->compute(dipole_ints);

    //    // transform SO dipole to MO dipole
    //    for(psi::SharedMatrix& dipole: dipole_ints){
    //        dipole->transform(ints_->Ca());
    //    }

    // prepare eigen vectors for ci_rdm
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
    for (size_t i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }

    // loop over states of the same symmetry
    trans_dipole_.clear();
    for (int A = 0; A < nroot_; ++A) {
        for (int B = A + 1; B < nroot_; ++B) {
            std::string trans_name = std::to_string(A) + " -> " + std::to_string(B);

            CI_RDMS ci_rdms(fci_ints_, determinant_, evecs, A, B);
            std::vector<double> opdm_a, opdm_b;
            ci_rdms.compute_1rdm(opdm_a, opdm_b);

            psi::SharedMatrix SOtransD =
                reformat_1rdm("SO transition density " + trans_name, opdm_a, true);
            SOtransD->back_transform(integral_->Ca());

            size_t nao = sotoao->coldim(0);
            psi::SharedMatrix AOtransD(
                new psi::Matrix("AO transition density " + trans_name, nao, nao));
            AOtransD->remove_symmetry(SOtransD, sotoao);

            std::vector<double> de(4, 0.0);
            for (int i = 0; i < 3; ++i) {
                de[i] = 2.0 * AOtransD->vector_dot(aodipole_ints[i]); // 2.0 for beta spin
                //                de[i] = 2.0 * MOtransD->vector_dot(dipole_ints[i]);
                de[3] += de[i] * de[i];
            }
            de[3] = sqrt(de[3]);

            trans_dipole_[trans_name] = de;

            if (de[3] > 1.0e-5) {
                outfile->Printf("\n  Transition dipole moments (a.u.) %s:  X: "
                                "%7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f",
                                trans_name.c_str(), de[0], de[1], de[2], de[3]);
            }
        }
    }
    outfile->Printf("\n");

    //    // use oeprop
    //    for(int A = 0; A < nroot_; ++A){
    //        CI_RDMS ci_rdms (fci_ints_,determinant_,evecs,0,A);
    //        std::vector<double> opdm_a (na_ * na_, 0.0);
    //        std::vector<double> opdm_b (na_ * na_, 0.0);
    //        ci_rdms.compute_1rdm(opdm_a, opdm_b);

    //        psi::SharedMatrix transD (new psi::Matrix("MO transition density 0 -> " +
    //        std::to_string(A), nmopi_, nmopi_));
    //        symmetrize_density(opdm_a, transD);
    //        transD->back_transform(ints_->Ca());

    //        boost::shared_ptr<OEProp> oe(new OEProp(reference_wavefunction_));
    //        oe->set_title("CAS TRANSITION");
    //        oe->add("TRANSITION_DIPOLE");
    //        oe->set_Da_so(transD);
    //        outfile->Printf( "  ==> Transition dipole moment computed with CAS
    //        <==\n\n");
    //        oe->compute();
    //    }
}

void FCI_MO::compute_oscillator_strength() {

    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::string irrep_symbol = ct.gamma(root_sym_).symbol();
    std::stringstream title;
    title << "Oscillator Strength (" << irrep_symbol << " -> " << irrep_symbol << ")";
    print_h2(title.str());
    outfile->Printf("\n  Only print nonzero (> 1.0e-5) elements.");

    // obtain the excitation energies map
    std::map<std::string, double> Exs;
    for (int A = 0; A < nroot_; ++A) {
        for (int B = A + 1; B < nroot_; ++B) {
            std::string trans_name = std::to_string(A) + " -> " + std::to_string(B);
            Exs[trans_name] = eigen_[B].second - eigen_[A].second;
        }
    }

    // loop over transition dipole
    for (const auto& x : trans_dipole_) {
        std::vector<double> oc(4, 0.0);
        for (int i = 0; i < 3; ++i) {
            double trdm = (x.second)[i];
            double ex = Exs[x.first];
            oc[i] = 2.0 / 3.0 * ex * trdm * trdm;
        }
        oc[3] = std::accumulate(oc.begin(), oc.end(), 0.0);

        if (oc[3] > 1.0e-5) {
            outfile->Printf("\n  Oscillator strength (a.u.) %s:  X: %7.4f  Y: "
                            "%7.4f  Z: %7.4f  Total: %7.4f",
                            x.first.c_str(), oc[0], oc[1], oc[2], oc[3]);
        }
    }
    outfile->Printf("\n");
}

std::map<std::string, std::vector<double>>
FCI_MO::compute_ref_relaxed_dm(const std::vector<double>& dm0, std::vector<BlockedTensor>& dm1,
                               std::vector<BlockedTensor>& dm2) {
    std::map<std::string, std::vector<double>> out;

    double dm0_sum = std::fabs(dm0[0]) + std::fabs(dm0[1]) + std::fabs(dm0[2]);
    std::vector<bool> do_dm;
    for (int z = 0; z < 3; ++z) {
        do_dm.push_back(std::fabs(dm0[z]) > 1.0e-12 ? true : false);
    }

    std::string pg = (psi::Process::environment.molecule()->point_group()->char_table()).symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi, const int& root, const int& irrep) {
        std::stringstream name_ss;
        name_ss << std::setw(2) << root << " " << std::setw(7) << multi_symbols_[multi - 1] << " "
                << std::setw(width) << irrep_symbols_[irrep];
        return name_ss.str();
    };

    // if SS, read from determinant_ and eigen_; otherwise, read from p_spaces_ and eigens_
    if (sa_info_.size() == 0) {
        std::string name = generate_name(multi_, root_, root_sym_);
        std::vector<double> dm(3, 0.0);

        // prepare CI_RDMS
        int dim = (eigen_[0].first)->dim();
        size_t eigen_size = eigen_.size();
        psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
        for (size_t i = 0; i < eigen_size; ++i) {
            evecs->set_column(0, i, (eigen_[i]).first);
        }

        // CI_RDMS for the targeted root
        CI_RDMS ci_rdms(fci_ints_, determinant_, evecs, root_, root_);

        if (dm0_sum > 1.0e-12) {
            // compute RDMS and put into BlockedTensor format
            ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
            ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

            // loop over directions
            for (int z = 0; z < 3; ++z) {
                if (do_dm[z]) {
                    dm[z] = ref_relaxed_dm_helper(dm0[z], dm1[z], dm2[z], D1, D2);
                }
            }
        }
        out[name] = dm;

    } else {
        int nentry = sa_info_.size();
        for (int n = 0; n < nentry; ++n) {
            // get current symmetry, multiplicity, nroots, weights
            int irrep, multi, nroots;
            std::vector<double> weights;
            std::tie(irrep, multi, nroots, weights) = sa_info_[n];

            // eigen vectors for current symmetry
            int dim = (eigens_[n][0].first)->dim();
            size_t eigen_size = eigens_[n].size();
            psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
            for (size_t i = 0; i < eigen_size; ++i) {
                evecs->set_column(0, i, (eigens_[n][i]).first);
            }

            // loop over nroots for current symmetry
            for (int i = 0; i < nroots; ++i) {
                std::string name = generate_name(multi, i, irrep);
                std::vector<double> dm(3, 0.0);

                CI_RDMS ci_rdms(fci_ints_, p_spaces_[n], evecs, i, i);

                if (dm0_sum > 1.0e-12) {
                    // compute RDMS and put into BlockedTensor format
                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                    // loop over directions
                    for (int z = 0; z < 3; ++z) {
                        if (do_dm[z]) {
                            dm[z] = ref_relaxed_dm_helper(dm0[z], dm1[z], dm2[z], D1, D2);
                        }
                    }
                }
                out[name] = dm;
            }
        }
    }
    return out;
}

std::map<std::string, std::vector<double>>
FCI_MO::compute_ref_relaxed_dm(const std::vector<double>& dm0, std::vector<BlockedTensor>& dm1,
                               std::vector<BlockedTensor>& dm2, std::vector<BlockedTensor>& dm3) {
    std::map<std::string, std::vector<double>> out;

    double dm0_sum = std::fabs(dm0[0]) + std::fabs(dm0[1]) + std::fabs(dm0[2]);
    std::vector<bool> do_dm;
    for (int z = 0; z < 3; ++z) {
        do_dm.push_back(std::fabs(dm0[z]) > 1.0e-12 ? true : false);
    }

    std::string pg = (psi::Process::environment.molecule()->point_group()->char_table()).symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi, const int& root, const int& irrep) {
        std::stringstream name_ss;
        name_ss << std::setw(2) << root << " " << std::setw(7) << multi_symbols_[multi - 1] << " "
                << std::setw(width) << irrep_symbols_[irrep];
        return name_ss.str();
    };

    // if SS, read from determinant_ and eigen_; otherwise, read from p_spaces_ and eigens_
    if (sa_info_.size() == 0) {
        std::string name = generate_name(multi_, root_, root_sym_);
        std::vector<double> dm(3, 0.0);

        // prepare CI_RDMS
        int dim = (eigen_[0].first)->dim();
        size_t eigen_size = eigen_.size();
        psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
        for (size_t i = 0; i < eigen_size; ++i) {
            evecs->set_column(0, i, (eigen_[i]).first);
        }

        // CI_RDMS for the targeted root
        CI_RDMS ci_rdms(fci_ints_, determinant_, evecs, root_, root_);

        if (dm0_sum > 1.0e-12) {
            // compute RDMS and put into BlockedTensor format
            ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
            ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);
            ambit::BlockedTensor D3 = compute_n_rdm(ci_rdms, 3);

            // loop over directions
            for (int z = 0; z < 3; ++z) {
                if (do_dm[z]) {
                    dm[z] = ref_relaxed_dm_helper(dm0[z], dm1[z], dm2[z], dm3[z], D1, D2, D3);
                }
            }
        }
        out[name] = dm;

    } else {
        int nentry = sa_info_.size();
        for (int n = 0; n < nentry; ++n) {
            // get current symmetry, multiplicity, nroots, weights
            int irrep, multi, nroots;
            std::vector<double> weights;
            std::tie(irrep, multi, nroots, weights) = sa_info_[n];

            // eigen vectors for current symmetry
            int dim = (eigens_[n][0].first)->dim();
            size_t eigen_size = eigens_[n].size();
            psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
            for (size_t i = 0; i < eigen_size; ++i) {
                evecs->set_column(0, i, (eigens_[n][i]).first);
            }

            // loop over nroots for current symmetry
            for (int i = 0; i < nroots; ++i) {
                std::string name = generate_name(multi, i, irrep);
                std::vector<double> dm(3, 0.0);

                CI_RDMS ci_rdms(fci_ints_, p_spaces_[n], evecs, i, i);

                if (dm0_sum > 1.0e-12) {
                    // compute RDMS and put into BlockedTensor format
                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);
                    ambit::BlockedTensor D3 = compute_n_rdm(ci_rdms, 3);

                    // loop over directions
                    for (int z = 0; z < 3; ++z) {
                        if (do_dm[z]) {
                            dm[z] =
                                ref_relaxed_dm_helper(dm0[z], dm1[z], dm2[z], dm3[z], D1, D2, D3);
                        }
                    }
                }
                out[name] = dm;
            }
        }
    }
    return out;
}

std::map<std::string, std::vector<double>>
FCI_MO::compute_ref_relaxed_osc(std::vector<BlockedTensor>& dm1, std::vector<BlockedTensor>& dm2) {
    std::map<std::string, std::vector<double>> out;

    std::string pg = (psi::Process::environment.molecule()->point_group()->char_table()).symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi0, const int& root0, const int& irrep0,
                             const int& multi1, const int& root1, const int& irrep1) {
        std::stringstream name_ss;
        name_ss << std::setw(2) << root0 << " " << std::setw(7) << multi_symbols_[multi0 - 1] << " "
                << std::setw(width) << irrep_symbols_[irrep0] << " -> " << std::setw(2) << root1
                << " " << std::setw(7) << multi_symbols_[multi1 - 1] << " " << std::setw(width)
                << irrep_symbols_[irrep1];
        return name_ss.str();
    };

    int nentry = sa_info_.size();
    for (int A = 0; A < nentry; ++A) {
        int irrep0, multi0, nroots0;
        std::vector<double> weights0;
        std::tie(irrep0, multi0, nroots0, weights0) = sa_info_[A];

        size_t ndets0 = (eigens_[A][0].first)->dim();
        psi::SharedMatrix evecs0(new psi::Matrix("evecs", ndets0, nroots0));
        for (int i = 0; i < nroots0; ++i) {
            evecs0->set_column(0, i, (eigens_[A][i]).first);
        }

        // oscillator strength of the same symmetry
        for (int i = 0; i < nroots0; ++i) {
            for (int j = i + 1; j < nroots0; ++j) {
                std::string name = generate_name(multi0, i, irrep0, multi0, j, irrep0);

                double Eex = eigens_[A][j].second - eigens_[A][i].second;
                std::vector<double> osc(3, 0.0);

                CI_RDMS ci_rdms(fci_ints_, p_spaces_[A], evecs0, i, j);

                ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                for (int z = 0; z < 3; ++z) {
                    double dm = ref_relaxed_dm_helper(0.0, dm1[z], dm2[z], D1, D2);
                    osc[z] = 2.0 / 3.0 * Eex * dm * dm;
                }

                out[name] = osc;
            }
        }

        // oscillator strength of different symmetry
        for (int B = A + 1; B < nentry; ++B) {
            int irrep1, multi1, nroots1;
            std::vector<double> weights1;
            std::tie(irrep1, multi1, nroots1, weights1) = sa_info_[B];

            // combine two eigen vectors
            size_t ndets1 = (eigens_[B][0].first)->dim();
            size_t ndets = ndets0 + ndets1;
            size_t nroots = nroots0 + nroots1;
            psi::SharedMatrix evecs(new psi::Matrix("evecs", ndets, nroots));

            for (int n = 0; n < nroots0; ++n) {
                psi::SharedVector evec0 = evecs0->get_column(0, n);
                psi::SharedVector evec(
                    new psi::Vector("combined evec0 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets0; ++i) {
                    evec->set(i, evec0->get(i));
                }
                evecs->set_column(0, n, evec);
            }

            for (int n = 0; n < nroots1; ++n) {
                psi::SharedVector evec1 = eigens_[B][n].first;
                psi::SharedVector evec(
                    new psi::Vector("combined evec1 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets1; ++i) {
                    evec->set(i + ndets0, evec1->get(i));
                }
                evecs->set_column(0, n + nroots0, evec);
            }

            // combine p_space
            std::vector<Determinant> p_space(p_spaces_[A]);
            std::vector<Determinant>& p_space1 = p_spaces_[B];
            p_space.insert(p_space.end(), p_space1.begin(), p_space1.end());

            for (int i = 0; i < nroots0; ++i) {
                for (int j = 0; j < nroots1; ++j) {
                    std::string name = generate_name(multi0, i, irrep0, multi1, j, irrep1);

                    double Eex = eigens_[B][j].second - eigens_[A][i].second;
                    std::vector<double> osc(3, 0.0);

                    CI_RDMS ci_rdms(fci_ints_, p_space, evecs, i, j + nroots0);

                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                    for (int z = 0; z < 3; ++z) {
                        double dm = ref_relaxed_dm_helper(0.0, dm1[z], dm2[z], D1, D2);
                        osc[z] = 2.0 / 3.0 * Eex * dm * dm;
                    }

                    out[name] = osc;
                }
            }
        }
    }

    return out;
}

std::map<std::string, std::vector<double>>
FCI_MO::compute_ref_relaxed_osc(std::vector<BlockedTensor>& dm1, std::vector<BlockedTensor>& dm2,
                                std::vector<BlockedTensor>& dm3) {
    std::map<std::string, std::vector<double>> out;

    std::string pg = (psi::Process::environment.molecule()->point_group()->char_table()).symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi0, const int& root0, const int& irrep0,
                             const int& multi1, const int& root1, const int& irrep1) {
        std::stringstream name_ss;
        name_ss << std::setw(2) << root0 << " " << std::setw(7) << multi_symbols_[multi0 - 1] << " "
                << std::setw(width) << irrep_symbols_[irrep0] << " -> " << std::setw(2) << root1
                << " " << std::setw(7) << multi_symbols_[multi1 - 1] << " " << std::setw(width)
                << irrep_symbols_[irrep1];
        return name_ss.str();
    };

    int nentry = sa_info_.size();
    for (int A = 0; A < nentry; ++A) {
        int irrep0, multi0, nroots0;
        std::vector<double> weights0;
        std::tie(irrep0, multi0, nroots0, weights0) = sa_info_[A];

        size_t ndets0 = (eigens_[A][0].first)->dim();
        psi::SharedMatrix evecs0(new psi::Matrix("evecs", ndets0, nroots0));
        for (int i = 0; i < nroots0; ++i) {
            evecs0->set_column(0, i, (eigens_[A][i]).first);
        }

        // oscillator strength of the same symmetry
        for (int i = 0; i < nroots0; ++i) {
            for (int j = i + 1; j < nroots0; ++j) {
                std::string name = generate_name(multi0, i, irrep0, multi0, j, irrep0);

                double Eex = eigens_[A][j].second - eigens_[A][i].second;
                std::vector<double> osc(3, 0.0);

                CI_RDMS ci_rdms(fci_ints_, p_spaces_[A], evecs0, i, j);

                ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);
                ambit::BlockedTensor D3 = compute_n_rdm(ci_rdms, 3);

                for (int z = 0; z < 3; ++z) {
                    double dm = ref_relaxed_dm_helper(0.0, dm1[z], dm2[z], dm3[z], D1, D2, D3);
                    osc[z] = 2.0 / 3.0 * Eex * dm * dm;
                }

                out[name] = osc;
            }
        }

        // oscillator strength of different symmetry
        for (int B = A + 1; B < nentry; ++B) {
            int irrep1, multi1, nroots1;
            std::vector<double> weights1;
            std::tie(irrep1, multi1, nroots1, weights1) = sa_info_[B];

            // combine two eigen vectors
            size_t ndets1 = (eigens_[B][0].first)->dim();
            size_t ndets = ndets0 + ndets1;
            size_t nroots = nroots0 + nroots1;
            psi::SharedMatrix evecs(new psi::Matrix("evecs", ndets, nroots));

            for (int n = 0; n < nroots0; ++n) {
                psi::SharedVector evec0 = evecs0->get_column(0, n);
                psi::SharedVector evec(
                    new psi::Vector("combined evec0 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets0; ++i) {
                    evec->set(i, evec0->get(i));
                }
                evecs->set_column(0, n, evec);
            }

            for (int n = 0; n < nroots1; ++n) {
                psi::SharedVector evec1 = eigens_[B][n].first;
                psi::SharedVector evec(
                    new psi::Vector("combined evec1 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets1; ++i) {
                    evec->set(i + ndets0, evec1->get(i));
                }
                evecs->set_column(0, n + nroots0, evec);
            }

            // combine p_space
            std::vector<Determinant> p_space(p_spaces_[A]);
            std::vector<Determinant>& p_space1 = p_spaces_[B];
            p_space.insert(p_space.end(), p_space1.begin(), p_space1.end());

            for (int i = 0; i < nroots0; ++i) {
                for (int j = 0; j < nroots1; ++j) {
                    std::string name = generate_name(multi0, i, irrep0, multi1, j, irrep1);

                    double Eex = eigens_[B][j].second - eigens_[A][i].second;
                    std::vector<double> osc(3, 0.0);

                    CI_RDMS ci_rdms(fci_ints_, p_space, evecs, i, j + nroots0);

                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);
                    ambit::BlockedTensor D3 = compute_n_rdm(ci_rdms, 3);

                    for (int z = 0; z < 3; ++z) {
                        double dm = ref_relaxed_dm_helper(0.0, dm1[z], dm2[z], dm3[z], D1, D2, D3);
                        osc[z] = 2.0 / 3.0 * Eex * dm * dm;
                    }

                    out[name] = osc;
                }
            }
        }
    }

    return out;
}

ambit::BlockedTensor FCI_MO::compute_n_rdm(CI_RDMS& cirdm, const int& order) {
    if (order < 1 || order > 3) {
        throw psi::PSIEXCEPTION("Cannot compute RDMs except 1, 2, 3.");
    }

    ambit::BlockedTensor out;
    if (order == 1) {
        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D1", spin_cases({"aa"}));
        cirdm.compute_1rdm(out.block("aa").data(), out.block("AA").data());
    } else if (order == 2) {
        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D2", spin_cases({"aaaa"}));
        cirdm.compute_2rdm(out.block("aaaa").data(), out.block("aAaA").data(),
                           out.block("AAAA").data());
    } else if (order == 3) {
        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D3", spin_cases({"aaaaaa"}));
        cirdm.compute_3rdm(out.block("aaaaaa").data(), out.block("aaAaaA").data(),
                           out.block("aAAaAA").data(), out.block("AAAAAA").data());
    }

    return out;
}

double FCI_MO::ref_relaxed_dm_helper(const double& dm0, BlockedTensor& dm1, BlockedTensor& dm2,
                                     BlockedTensor& D1, BlockedTensor& D2) {
    double dm_out = dm0;

    dm_out += dm1["uv"] * D1["uv"];
    dm_out += dm1["UV"] * D1["UV"];
    dm_out += 0.25 * dm2["uvxy"] * D2["uvxy"];
    dm_out += 0.25 * dm2["UVXY"] * D2["UVXY"];
    dm_out += dm2["uVxY"] * D2["uVxY"];

    return dm_out;
}

double FCI_MO::ref_relaxed_dm_helper(const double& dm0, BlockedTensor& dm1, BlockedTensor& dm2,
                                     BlockedTensor& dm3, BlockedTensor& D1, BlockedTensor& D2,
                                     BlockedTensor& D3) {
    double dm_out = dm0;

    dm_out += dm1["uv"] * D1["uv"];
    dm_out += dm1["UV"] * D1["UV"];

    dm_out += 0.25 * dm2["uvxy"] * D2["uvxy"];
    dm_out += 0.25 * dm2["UVXY"] * D2["UVXY"];
    dm_out += dm2["uVxY"] * D2["uVxY"];

    dm_out += 1.0 / 36.0 * dm3["uvwxyz"] * D3["uvwxyz"];
    dm_out += 1.0 / 36.0 * dm3["UVWXYZ"] * D3["UVWXYZ"];
    dm_out += 0.25 * dm3["uvWxyZ"] * D3["uvWxyZ"];
    dm_out += 0.25 * dm3["uVWxYZ"] * D3["uVWxYZ"];

    return dm_out;
}

d3 FCI_MO::compute_orbital_extents() {

    // compute AO quadrupole integrals
    std::shared_ptr<psi::BasisSet> basisset = integral_->basisset();
    std::shared_ptr<IntegralFactory> ints = std::shared_ptr<IntegralFactory>(
        new IntegralFactory(basisset, basisset, basisset, basisset));

    std::vector<psi::SharedMatrix> ao_Qpole;
    for (const std::string& direction : {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}) {
        std::string name = "AO Quadrupole" + direction;
        ao_Qpole.push_back(std::make_shared<psi::Matrix>(name, basisset->nbf(), basisset->nbf()));
    }
    std::shared_ptr<OneBodyAOInt> aoqOBI(ints->ao_quadrupole());
    aoqOBI->compute(ao_Qpole);

    // orbital coefficients arranged by orbital energies
    psi::SharedMatrix Ca_ao = integral_->Ca_subset("AO");
    int nao = Ca_ao->nrow();
    int nmo = Ca_ao->ncol();

    std::vector<psi::SharedVector> quadrupole;
    quadrupole.push_back(psi::SharedVector(new psi::Vector("Orbital Quadrupole XX", nmo)));
    quadrupole.push_back(psi::SharedVector(new psi::Vector("Orbital Quadrupole YY", nmo)));
    quadrupole.push_back(psi::SharedVector(new psi::Vector("Orbital Quadrupole ZZ", nmo)));

    for (int i = 0; i < nmo; ++i) {
        double sumx = 0.0, sumy = 0.0, sumz = 0.0;
        for (int k = 0; k < nao; ++k) {
            for (int l = 0; l < nao; ++l) {
                double tmp = Ca_ao->get(0, k, i) * Ca_ao->get(0, l, i);
                sumx += ao_Qpole[0]->get(0, k, l) * tmp;
                sumy += ao_Qpole[3]->get(0, k, l) * tmp;
                sumz += ao_Qpole[5]->get(0, k, l) * tmp;
            }
        }

        quadrupole[0]->set(0, i, std::fabs(sumx));
        quadrupole[1]->set(0, i, std::fabs(sumy));
        quadrupole[2]->set(0, i, std::fabs(sumz));
    }

    psi::SharedVector epsilon_a = scf_info_->epsilon_a();
    std::vector<std::tuple<double, int, int>> metric;
    for (int h = 0; h < epsilon_a->nirrep(); ++h) {
        for (int i = 0; i < epsilon_a->dimpi()[h]; ++i) {
            metric.push_back(std::tuple<double, int, int>(epsilon_a->get(h, i), i, h));
        }
    }
    std::sort(metric.begin(), metric.end());

    // initialize vector saving current orbital extents
    d3 orb_extents = std::vector<d2>(nirrep_, d2());
    for (int h = 0; h < nirrep_; ++h) {
        size_t na = actv_dim_[h];
        if (na == 0)
            continue;
        orb_extents[h] = d2(na, d1());
    }

    for (int n = 0, size = metric.size(); n < size; ++n) {
        double epsilon;
        int i, h;
        std::tie(epsilon, i, h) = metric[n];

        int offset = frzc_dim_[h] + core_dim_[h];
        if (i < offset || i >= offset + actv_dim_[h])
            continue;

        double xx = quadrupole[0]->get(0, n), yy = quadrupole[1]->get(0, n),
               zz = quadrupole[2]->get(0, n);
        orb_extents[h][i - offset] = {xx, yy, zz};
    }

    // find the diffused orbital index (active zero based)
    if (ipea_ != "NONE") {
        bool found = false;

        diffused_orbs_.clear();
        size_t offset = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < actv_dim_[h]; ++i) {
                double orbext = orb_extents[h][i][0] + orb_extents[h][i][1] + orb_extents[h][i][2];

                if (orbext > 1.0e6) {
                    diffused_orbs_.push_back(i + offset);

                    if (h == 0) {
                        idx_diffused_ = i; // totally symmetric diffused orbital
                        found = true;
                    }
                }
            }
            offset += actv_dim_[h];
        }

        if (!found) {
            outfile->Printf("\n  Totally symmetric diffused orbital is not found.");
            outfile->Printf("\n  Make sure a diffused s function is added to the basis.");
            throw psi::PSIEXCEPTION("Totally symmetric diffused orbital is not found.");
        }
    }

    return orb_extents;
}

std::vector<Reference> FCI_MO::reference(const std::vector<std::pair<size_t, size_t>>& root_list) {

    std::vector<Reference> refs;
    // if ((options_->psi_options())["AVG_STATE"].size() != 0) {
    //     Reference ref;
    //     compute_sa_ref(max_rdm_);
    //     ref.set_Eref(Eref_);

    //     if (max_rdm_ > 0) {
    //         ref.set_L1a(L1a_);
    //         ref.set_L1b(L1b_);
    //     }

    //     if (max_rdm_ > 1) {
    //         ref.set_L2aa(L2aa_);
    //         ref.set_L2ab(L2ab_);
    //         ref.set_L2bb(L2bb_);
    //     }

    //     if (max_rdm_ > 2 && (options_->get_str("THREEPDC") != "ZERO")) {
    //         ref.set_L3aaa(L3aaa_);
    //         ref.set_L3aab(L3aab_);
    //         ref.set_L3abb(L3abb_);
    //         ref.set_L3bbb(L3bbb_);
    //     }
    //     refs.push_back(ref);
    // } else {

    for (auto& roots : root_list) {
        compute_ref(max_rdm_level_, roots.first, roots.second);

        if (max_rdm_level_ == 1) {
            refs.emplace_back(L1a_, L1b_);
        }

        if (max_rdm_level_ == 2) {
            refs.emplace_back(L1a_, L1b_, L2aa_, L2ab_, L2bb_);
        }

        if (max_rdm_level_ == 3 && (options_->get_str("THREEPDC") != "ZERO")) {
            refs.emplace_back(L1a_, L1b_, L2aa_, L2ab_, L2bb_, L3aaa_, L3aab_, L3abb_, L3bbb_);
        }
    }
    //}
    return refs;
}

std::vector<Reference> FCI_MO::densities(const std::vector<std::pair<size_t, size_t>>& root_list,
                                         std::shared_ptr<ActiveSpaceMethod> method2,
                                         int max_rdm_level) {
    // TODO : add code to handle transition density matrices (Francesco)
    std::vector<Reference> refs;
    for (auto& roots : root_list) {
        compute_ref(max_rdm_level, roots.first, roots.second);

        if (max_rdm_level_ == 1) {
            refs.emplace_back(L1a_, L1b_);
        }

        if (max_rdm_level_ == 2) {
            refs.emplace_back(L1a_, L1b_, L2aa_, L2ab_, L2bb_);
        }

        if (max_rdm_level_ == 3 && (options_->get_str("THREEPDC") != "ZERO")) {
            refs.emplace_back(L1a_, L1b_, L2aa_, L2ab_, L2bb_, L3aaa_, L3aab_, L3abb_, L3bbb_);
        }
    }
    return refs;
}

void FCI_MO::compute_ref(const int& level, size_t root1, size_t root2) {
    timer_on("Compute Ref");
    if (!quiet_) {
        //      print_h2("Compute State-Specific Cumulants");
        outfile->Printf("\n  Computing (%d,%d) RDMs", root1, root2);
    }

    // prepare eigen vectors for ci_rdms
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
    for (size_t i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }

    // compute 1-RDM
    auto D1 = compute_n_rdm(determinant_, evecs, 1, root1, root2, root_sym_, multi_, false);
    L1a_ = D1[0];
    L1b_ = D1[1];

    // compute 2-RDM
    if (level >= 2) {
        auto D2 = compute_n_rdm(determinant_, evecs, 2, root1, root2, root_sym_, multi_, false);
        L2aa_ = D2[0];
        L2ab_ = D2[1];
        L2bb_ = D2[2];
        //        add_wedge_cu2(L1a_, L1b_, L2aa_, L2ab_, L2bb_);
    }

    // compute 3-RDM
    std::string threepdc = options_->get_str("THREEPDC");
    if (threepdc != "ZERO" && level >= 3) {
        if (threepdc == "MK") {
            auto D3 = compute_n_rdm(determinant_, evecs, 3, root1, root2, root_sym_, multi_, false);
            L3aaa_ = D3[0];
            L3aab_ = D3[1];
            L3abb_ = D3[2];
            L3bbb_ = D3[3];
        } else {
            L3aaa_ =
                ambit::Tensor::build(ambit::CoreTensor, "L3aaa", std::vector<size_t>(6, nactv_));
            L3aab_ =
                ambit::Tensor::build(ambit::CoreTensor, "L3aab", std::vector<size_t>(6, nactv_));
            L3abb_ =
                ambit::Tensor::build(ambit::CoreTensor, "L3abb", std::vector<size_t>(6, nactv_));
            L3bbb_ =
                ambit::Tensor::build(ambit::CoreTensor, "L3bbb", std::vector<size_t>(6, nactv_));
        }
        //        add_wedge_cu3(L1a_, L1b_, L2aa_, L2ab_, L2bb_, L3aaa_, L3aab_, L3abb_, L3bbb_);
    }

    timer_off("Compute Ref");
}

void FCI_MO::add_wedge_cu2(const ambit::Tensor& L1a, const ambit::Tensor& L1b, ambit::Tensor& L2aa,
                           ambit::Tensor& L2ab, ambit::Tensor& L2bb) {
    std::string job_name = "add_wedge_cu2";
    outfile->Printf("\n  Adding wedge product for 2-cumulants ... ");
    timer_on(job_name);
    local_timer timer;

    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");

    outfile->Printf("Done. Timing %15.6f s", timer.get());
    timer_off(job_name);
}

void FCI_MO::add_wedge_cu3(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                           const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                           const ambit::Tensor& L2bb, ambit::Tensor& L3aaa, ambit::Tensor& L3aab,
                           ambit::Tensor& L3abb, ambit::Tensor& L3bbb) {
    std::string job_name = "add_wedge_cu3";
    outfile->Printf("\n  Adding wedge product for 3-cumulants ... ");
    timer_on(job_name);
    local_timer timer;

    // aaa
    L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
    L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
    L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

    L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
    L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
    L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

    L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
    L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
    L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

    L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
    L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
    L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

    L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
    L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
    L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");

    // aab
    L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
    L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

    // abb
    L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
    L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

    // bbb
    L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
    L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
    L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

    L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
    L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
    L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

    L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
    L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
    L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

    L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
    L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
    L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

    L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
    L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
    L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");

    outfile->Printf("Done. Timing %15.6f s", timer.get());
    timer_off(job_name);
}

double FCI_MO::compute_sa_energy() {
    // averaged energy
    double Ecas_sa = 0.0;

    // clear eigen values, eigen vectors and determinants
    eigens_.clear();
    p_spaces_.clear();

    // loop over all averaged states
    int nstates = 0;
    for (const auto& info : sa_info_) {
        // get current symmetry, multiplicity, nroots, weights
        int irrep, multi, nroots;
        std::vector<double> weights;
        std::tie(irrep, multi, nroots, weights) = info;
        nstates += nroots;

        root_sym_ = irrep;
        multi_ = multi;
        nroot_ = nroots;
        root_ = nroot_ - 1; // not necessary

        // form determinants
        form_p_space();
        p_spaces_.push_back(determinant_);

        // diagonalize the CASCI Hamiltonian
        eigen_.clear();
        Diagonalize_H(determinant_, multi_, nroot_, eigen_);
        eigens_.push_back(eigen_);

        // print CI vectors in eigen_
        int eigen_size = eigen_.size();
        if (nroot_ > eigen_size) {
            outfile->Printf("\n  Too many roots of interest!");
            std::string be = (eigen_size > 1) ? "are" : "is";
            std::string plural = (eigen_size > 1) ? "roots" : "root";
            outfile->Printf("\n  There %s only %3d %s that satisfy the condition!", be.c_str(),
                            eigen_size, plural.c_str());
            outfile->Printf("\n  Check root_sym, multi, and number of determinants.");
            throw psi::PSIEXCEPTION("Too many roots of interest.");
        }
        print_CI(nroot_, options_->get_double("FCIMO_PRINT_CIVEC"), eigen_, determinant_);

        // weight energies
        for (int i = 0; i < nroots; ++i) {
            Ecas_sa += weights[i] * eigen_[i].second;
        }

        // compute dipole moments
        compute_permanent_dipole();

        // compute oscillator strength
        if (nroot_ > 1) {
            compute_transition_dipole();
            compute_oscillator_strength();
        }
    }                     // end looping over all averaged states
    eigen_.clear();       // make sure other code use eigens_ for state average
    determinant_.clear(); // make sure other code use p_spaces_ for state average
    outfile->Printf("\n  Total Energy (averaged over %d states): %20.15f\n", nstates, Ecas_sa);

    Eref_ = Ecas_sa;
    psi::Process::environment.globals["CURRENT ENERGY"] = Ecas_sa;
    return Ecas_sa;
}

void FCI_MO::xms_rotate_civecs() {
    if (eigens_.size() != sa_info_.size()) {
        throw psi::PSIEXCEPTION(
            "Cannot do XMS rotation due to inconsistent size. Is CASCI computed?");
    }
    int nentry = eigens_.size();

    // title
    print_h2("XMS Rotation of All CI Vectors");

    // form averaged density (all roots equal weight)
    int total_nroots = 0;
    for (int n = 0; n < nentry; ++n) {
        total_nroots += std::get<2>(sa_info_[n]);
    }
    double w = 1.0 / total_nroots;
    outfile->Printf("\n  Set equal states weights to %.4f = 1/%d.", w, total_nroots);

    auto sa_info0 = sa_info();
    sa_info_.resize(nentry);
    for (int n = 0; n < nentry; ++n) {
        int irrep, multi, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info0[n];
        sa_info_[n] = std::make_tuple(irrep, multi, nroots, std::vector<double>(nroots, w));
    }
    // compute_sa_ref(1);
    safe_to_read_density_files_ = false;
    sa_info_ = sa_info0;

    // form averaged Fock matrix
    outfile->Printf("  Form averaged Fock matrix.");
    ambit::Tensor Fa = ambit::Tensor::build(CoreTensor, "Fa", {nactv_, nactv_});
    ambit::Tensor Fb = ambit::Tensor::build(CoreTensor, "Fb", {nactv_, nactv_});

    Fa.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = integral_->oei_a(nu, nv);
    });

    Fb.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = integral_->oei_b(nu, nv);
    });

    ambit::Tensor I = ambit::Tensor::build(CoreTensor, "Identity", {ncore_, ncore_});
    for (size_t m = 0; m < ncore_; ++m) {
        I.data()[m * ncore_ + m] = 1.0;
    }

    ambit::Tensor V;
    V = integral_->aptei_aa_block(actv_mos_, core_mos_, actv_mos_, core_mos_);
    Fa("uv") += V("umvn") * I("mn");

    V = integral_->aptei_ab_block(actv_mos_, core_mos_, actv_mos_, core_mos_);
    Fa("uv") += V("umvn") * I("mn");

    V = integral_->aptei_ab_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    Fb("uv") += V("munv") * I("mn");

    V = integral_->aptei_bb_block(actv_mos_, core_mos_, actv_mos_, core_mos_);
    Fb("uv") += V("umvn") * I("mn");

    V = integral_->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    Fa("uv") += V("uxvy") * L1a_("xy");

    V = integral_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    Fa("uv") += V("uxvy") * L1b_("xy");
    Fb("uv") += V("xuyv") * L1a_("xy");

    V = integral_->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_);
    Fb("uv") += V("uxvy") * L1b_("xy");

    // XMS rotation for all symmetries
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info_[n];

        outfile->Printf("\n  XMS Rotation for %s %s.\n", multi_symbols_[multi - 1].c_str(),
                        irrep_symbols_[irrep].c_str());

        // put eigen vectors of current symmetry entry to psi::SharedMatrix form
        auto& eigen = eigens_[n];
        int dim = (eigen[0].first)->dim();
        psi::SharedMatrix civecs(new psi::Matrix("ci vecs", dim, nroots));
        for (int i = 0; i < nroots; ++i) {
            civecs->set_column(0, i, (eigen[i]).first);
        }

        // compute averaged Fock matrix between states <M|F|N>
        psi::SharedMatrix rcivecs = xms_rotate_this_civecs(p_spaces_[n], civecs, Fa, Fb);

        // put in eigens_
        for (int i = 0; i < nroots; ++i) {
            eigens_[n][i] =
                std::make_pair<psi::SharedVector, double>(rcivecs->get_column(0, i), 0.0);
        }
    }
}

psi::SharedMatrix FCI_MO::xms_rotate_this_civecs(const det_vec& p_space, psi::SharedMatrix civecs,
                                                 ambit::Tensor Fa, ambit::Tensor Fb) {
    int nroots = civecs->ncol();
    outfile->Printf("\n");
    psi::SharedMatrix Fock(new psi::Matrix("Fock <M|F|N>", nroots, nroots));

    for (int M = 0; M < nroots; ++M) {
        for (int N = M; N < nroots; ++N) {

            // compute transition density
            ambit::Tensor Da = ambit::Tensor::build(CoreTensor, "Da", {nactv_, nactv_});
            ambit::Tensor Db = ambit::Tensor::build(CoreTensor, "Da", {nactv_, nactv_});
            CI_RDMS ci_rdms(fci_ints_, p_space, civecs, M, N);
            ci_rdms.compute_1rdm(Da.data(), Db.data());

            // compute Fock elements
            double F_MN = 0.0;
            F_MN += Da("uv") * Fa("vu");
            F_MN += Db("UV") * Fb("VU");
            Fock->set(M, N, F_MN);
            if (M != N) {
                Fock->set(N, M, F_MN);
            }
        }
    }
    Fock->print();

    // diagonalize Fock
    psi::SharedMatrix Fevec(new psi::Matrix("Fock Evec", nroots, nroots));
    psi::SharedVector Feval(new psi::Vector("Fock Eval", nroots));
    Fock->diagonalize(Fevec, Feval);
    Fevec->eivprint(Feval);

    // Rotate CI vectors
    psi::SharedMatrix rcivecs(civecs->clone());
    rcivecs->zero();
    rcivecs->gemm(false, false, 1.0, civecs, Fevec, 0.0);

    return rcivecs;
}

// void FCI_MO::compute_sa_ref(const int& level) {
//    timer_on("Compute SA Ref");
//    if (!quiet_) {
//        print_h2("Compute State-Averaged Cumulants");
//    }
//
//    // prepare averaged densities
//    L1a_ = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nactv_, nactv_});
//    L1b_ = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nactv_, nactv_});
//
//    if (level >= 2) {
//        L2aa_ = ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nactv_, nactv_, nactv_, nactv_});
//        L2ab_ = ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nactv_, nactv_, nactv_, nactv_});
//        L2bb_ = ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nactv_, nactv_, nactv_, nactv_});
//    }
//
//    std::string threepdc = options_->get_str("THREEPDC");
//    if (level >= 3 && threepdc != "ZERO") {
//        L3aaa_ = ambit::Tensor::build(ambit::CoreTensor, "L3aaa", std::vector<size_t>(6, nactv_));
//        L3aab_ = ambit::Tensor::build(ambit::CoreTensor, "L3aab", std::vector<size_t>(6, nactv_));
//        L3abb_ = ambit::Tensor::build(ambit::CoreTensor, "L3abb", std::vector<size_t>(6, nactv_));
//        L3bbb_ = ambit::Tensor::build(ambit::CoreTensor, "L3bbb", std::vector<size_t>(6, nactv_));
//    }
//
//    // function that scale pdm by w and add scaled pdm to sa_pdm
//    auto scale_add = [](std::vector<double>& sa_pdm, std::vector<double>& pdm, const double& w) {
//        std::for_each(pdm.begin(), pdm.end(), [&](double& v) { v *= w; });
//        std::transform(sa_pdm.begin(), sa_pdm.end(), pdm.begin(), sa_pdm.begin(),
//                       std::plus<double>());
//    };
//
//    // save state-specific density to disk for DWMS-DSRG-PT
//    bool do_disk = options_->get_str("JOB_TYPE") == "DWMS-DSRGPT2";
//
//    for (size_t n = 0, nentry = sa_info_.size(); n < nentry; ++n) {
//        // get current nroots and weights
//        int nroots, irrep, multi;
//        std::vector<double> weights;
//        std::tie(irrep, multi, nroots, weights) = sa_info_[n];
//
//        // prepare eigen vectors for current symmetry
//        int dim = (eigens_[n][0].first)->dim();
//        size_t eigen_size = eigens_[n].size();
//        psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
//        for (size_t i = 0; i < eigen_size; ++i) {
//            evecs->set_column(0, i, (eigens_[n][i]).first);
//        }
//
//        for (int i = 0; i < nroots; ++i) {
//            double weight = weights[i];
//
//            // compute 1-RDMs
//            auto D1 = compute_n_rdm(p_spaces_[n], evecs, 1, i, i, irrep, multi, do_disk);
//            scale_add(L1a_.data(), D1[0].data(), weight);
//            scale_add(L1b_.data(), D1[1].data(), weight);
//
//            // compute 2-RDMs
//            if (level >= 2) {
//                auto D2 = compute_n_rdm(p_spaces_[n], evecs, 2, i, i, irrep, multi, do_disk);
//                scale_add(L2aa_.data(), D2[0].data(), weight);
//                scale_add(L2ab_.data(), D2[1].data(), weight);
//                scale_add(L2bb_.data(), D2[2].data(), weight);
//            }
//
//            if (level >= 3 && threepdc == "MK") {
//                auto D3 = compute_n_rdm(p_spaces_[n], evecs, 3, i, i, irrep, multi, do_disk);
//                scale_add(L3aaa_.data(), D3[0].data(), weight);
//                scale_add(L3aab_.data(), D3[1].data(), weight);
//                scale_add(L3abb_.data(), D3[2].data(), weight);
//                scale_add(L3bbb_.data(), D3[3].data(), weight);
//            }
//        }
//    } // end looping over all averaged states
//
//    safe_to_read_density_files_ = true;
//
//    // compute 2-cumulants and fill in L2 tensors
//    if (level >= 2) {
//        add_wedge_cu2(L1a_, L1b_, L2aa_, L2ab_, L2bb_);
//    }
//
//    // compute 3-cumulants and fill in L3 tensors
//    if (level >= 3 && threepdc != "ZERO") {
//        add_wedge_cu3(L1a_, L1b_, L2aa_, L2ab_, L2bb_, L3aaa_, L3aab_, L3abb_, L3bbb_);
//    }
//
//    timer_off("Compute SA Ref");
//}

bool FCI_MO::check_density_files(int rdm_level, int irrep, int multi, int root1, int root2) {
    auto filenames = density_filenames_generator(rdm_level, irrep, multi, root1, root2);

    bool out = true;
    for (const std::string& filename : filenames) {
        if (density_files_.find(filename) == density_files_.end()) {
            out = false;
            break;
        }
    }

    return out;
}

std::vector<std::string> FCI_MO::density_filenames_generator(int rdm_level, int irrep, int multi,
                                                             int root1, int root2) {
    std::vector<std::string> out, spins;
    out.reserve(rdm_level + 1);

    if (rdm_level == 1) {
        spins = std::vector<std::string>{"a", "b"};
    } else if (rdm_level == 2) {
        spins = std::vector<std::string>{"aa", "ab", "bb"};
    } else if (rdm_level == 3) {
        spins = std::vector<std::string>{"aaa", "aab", "abb", "bbb"};
    } else {
        throw psi::PSIEXCEPTION("RDM level > 3 is not supported.");
    }

    std::string path0 = PSIOManager::shared_object()->get_default_path() + "psi." +
                        std::to_string(getpid()) + "." +
                        psi::Process::environment.molecule()->name();
    std::string level = std::to_string(rdm_level);
    std::string name0 = (root1 == root2) ? level + "RDM" : level + "TrDM";

    for (const std::string& spin : spins) {
        std::string name = name0 + spin;
        std::string path = path0;
        std::vector<std::string> components = {name,
                                               std::to_string(root1),
                                               std::to_string(root2),
                                               std::to_string(irrep),
                                               std::to_string(multi),
                                               "bin"};
        for (const std::string& str : components) {
            path += "." + str;
        }
        out.push_back(path);
    }

    return out;
}

void FCI_MO::remove_density_files(int rdm_level, int irrep, int multi, int root1, int root2) {
    auto fullnames = density_filenames_generator(rdm_level, irrep, multi, root1, root2);
    for (const std::string& filename : fullnames) {
        density_files_.erase(filename);
        if (remove(filename.c_str()) != 0) {
            std::stringstream ss;
            ss << "Error deleting file " << filename << ": No such file or directory";
            throw psi::PSIEXCEPTION(ss.str());
        }
    }

    std::string level = std::to_string(rdm_level);
    std::string name = (root1 == root2) ? level + "RDM" : level + "TrDM";
    outfile->Printf("\n  Deleted files from disk for %s(%d-%d) of %s %s.", name.c_str(), root1,
                    root2, multi_symbols_[multi - 1].c_str(), irrep_symbols_[irrep].c_str());
}

void FCI_MO::clean_all_density_files() {
    for (const std::string& filename : density_files_) {
        if (remove(filename.c_str()) != 0) {
            std::stringstream ss;
            ss << "Error deleting file " << filename << ": No such file or directory";
            throw psi::PSIEXCEPTION(ss.str());
        }
    }
    density_files_.clear();
}

void FCI_MO::set_sa_info(const std::vector<std::tuple<int, int, int, std::vector<double>>>& info) {
    size_t nentry = info.size();
    if (sa_info_.size() == nentry) {
        for (size_t n = 0; n < nentry; ++n) {
            int multi, irrep, nroots;
            std::vector<double> weights;
            std::tie(irrep, multi, nroots, weights) = info[n];
            if (static_cast<size_t>(nroots) != weights.size()) {
                outfile->Printf("\n  Irrep: %d, Multi: %d, Nroots: %d, Nweights: %d", irrep, multi,
                                nroots, weights.size());
                psi::PSIEXCEPTION(
                    "Cannot set sa_info of FCI_MO: mismatching nroot and weights size.");
            }
        }
        sa_info_ = info;
    } else {
        throw psi::PSIEXCEPTION("Cannot set sa_info of FCI_MO: mismatching number of SA entries.");
    }
}

void FCI_MO::set_eigens(const std::vector<vector<pair<psi::SharedVector, double>>>& eigens) {
    size_t nentry = sa_info_.size();
    if (eigens.size() == nentry) {
        for (size_t n = 0; n < nentry; ++n) {
            int ne = std::get<2>(sa_info_[n]);
            if (eigens[n].size() != static_cast<size_t>(ne)) {
                outfile->Printf("\n  Entry %d: expected size %d, got %d", n, ne, eigens[n].size());
            }
        }
        eigens_ = eigens;
        safe_to_read_density_files_ = false;
        clean_all_density_files();
    } else {
        throw psi::PSIEXCEPTION("Cannot set eigens of FCI_MO: mismatching number of SA entries.");
    }
}

std::vector<ambit::Tensor> FCI_MO::compute_n_rdm(const vecdet& p_space, psi::SharedMatrix evecs,
                                                 int rdm_level, int root1, int root2, int irrep,
                                                 int multi, bool disk) {
    if (rdm_level > 3 || rdm_level < 1) {
        throw psi::PSIEXCEPTION("Incorrect RDM_LEVEL. Check your code!");
    }

    std::string job_name = root1 == root2 ? "RDM" : "TrDM";
    job_name = std::to_string(rdm_level) + job_name;
    timer_on(job_name);
    outfile->Printf("\n  Computing %5s (%d-%d) of %s %s ... ", job_name.c_str(), root1, root2,
                    multi_symbols_[multi - 1].c_str(), irrep_symbols_[irrep].c_str());
    local_timer timer;

    std::vector<std::string> names;
    if (rdm_level == 1) {
        names = std::vector<std::string>{"D1a", "D1b"};
    } else if (rdm_level == 2) {
        names = std::vector<std::string>{"D2aa", "D2ab", "D2bb"};
    } else if (rdm_level == 3) {
        names = std::vector<std::string>{"D3aaa", "D3aab", "D3abb", "D3bbb"};
    }

    int ntensors = rdm_level + 1;

    std::vector<ambit::Tensor> out;
    out.reserve(ntensors);
    for (int i = 0; i < ntensors; ++i) {
        out.emplace_back(ambit::Tensor::build(ambit::CoreTensor, names[i],
                                              std::vector<size_t>(2 * rdm_level, nactv_)));
    }

    auto filenames = density_filenames_generator(rdm_level, irrep, multi, root1, root2);
    bool files_exist = check_density_files(rdm_level, irrep, multi, root1, root2);

    if (safe_to_read_density_files_ && files_exist) {
        outfile->Printf("Reading ... ");
        for (int i = 0; i < ntensors; ++i) {
            read_disk_vector_double(filenames[i], out[i].data());
        }
    } else {
        CI_RDMS ci_rdms(fci_ints_, p_space, evecs, root1, root2);

        if (rdm_level == 1) {
            ci_rdms.compute_1rdm(out[0].data(), out[1].data());
        } else if (rdm_level == 2) {
            ci_rdms.compute_2rdm(out[0].data(), out[1].data(), out[2].data());
        } else if (rdm_level == 3) {
            ci_rdms.compute_3rdm(out[0].data(), out[1].data(), out[2].data(), out[3].data());
        }

        if (files_exist) {
            remove_density_files(rdm_level, irrep, multi, root1, root2);
        }

        if (disk) {
            outfile->Printf("Writing ... ");
            for (int i = 0; i < ntensors; ++i) {
                write_disk_vector_double(filenames[i], out[i].data());
                density_files_.insert(filenames[i]);
            }
        }
    }

    outfile->Printf("Done. Timing %15.6f s", timer.get());
    timer_off(job_name);
    return out;
}

Reference FCI_MO::transition_reference(int root1, int root2, bool multi_state, int entry,
                                       int max_level, bool do_cumulant, bool disk) {
    if (max_level > 3 || max_level < 1) {
        throw psi::PSIEXCEPTION("Max RDM level > 3 or < 1 is not available.");
    }

    int irrep = root_sym_;
    int multi = multi_;
    if (multi_state) {
        int nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info_[entry];
        if (root1 >= nroots || root2 >= nroots) {
            throw psi::PSIEXCEPTION("Root label overflows.");
        }
    }

    std::string job_type = do_cumulant ? "PDC" : "RDM";
    if (root1 != root2) {
        job_type = "TrDM";
        if (do_cumulant) {
            throw psi::PSIEXCEPTION("Cannot compute transition cumulants.");
        }
    }

    vecdet& p_space = multi_state ? p_spaces_[entry] : determinant_;
    std::vector<pair<psi::SharedVector, double>>& eigen = multi_state ? eigens_[entry] : eigen_;

    // prepare eigenvectors
    size_t dim = p_space.size();
    size_t eigen_size = eigen.size();
    psi::SharedMatrix evecs(new psi::Matrix("evecs", dim, eigen_size));
    for (size_t i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen[i]).first);
    }

    if (max_level == 1) {
        auto D1 = compute_n_rdm(p_space, evecs, 1, root1, root2, irrep, multi, disk);
        Reference ref(D1[0], D1[1]);
        return ref;
    } else if (max_level == 2) {
        auto D1 = compute_n_rdm(p_space, evecs, 1, root1, root2, irrep, multi, disk);
        auto D2 = compute_n_rdm(p_space, evecs, 2, root1, root2, irrep, multi, disk);
        Reference ref(D1[0], D1[1], D2[0], D2[1], D2[2]);
        return ref;
    } else if (max_level == 3) {
        auto D1 = compute_n_rdm(p_space, evecs, 1, root1, root2, irrep, multi, disk);
        auto D2 = compute_n_rdm(p_space, evecs, 2, root1, root2, irrep, multi, disk);
        auto D3 = compute_n_rdm(p_space, evecs, 3, root1, root2, irrep, multi, disk);
        Reference ref(D1[0], D1[1], D2[0], D2[1], D2[2], D3[0], D3[1], D3[2], D3[3]);
        return ref;
    } else {
        throw psi::PSIEXCEPTION("Max RDM level > 3 or < 1 is not available.");
    }
}

void FCI_MO::print_det(const vecdet& dets) {
    print_h2("Determinants |alpha|beta>");
    for (const Determinant& x : dets) {
        outfile->Printf("\n  %s", x.str().c_str());
    }
    outfile->Printf("\n");
}

void FCI_MO::print_occupation_strings_perirrep(std::string name,
                                               const vector<vector<vector<bool>>>& string) {
    print_h2(name);
    for (size_t i = 0; i != string.size(); ++i) {
        if (string[i].size() != 0) {
            outfile->Printf("\n  symmetry = %lu \n", i);
        }
        for (size_t j = 0; j != string[i].size(); ++j) {
            outfile->Printf("    ");
            for (bool b : string[i][j]) {
                outfile->Printf("%d ", b);
            }
            outfile->Printf("\n");
        }
    }
}
} // namespace forte
