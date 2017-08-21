/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "determinant_hashvector.h"
#include "fci/fci_vector.h"
#include "fci_mo.h"
#include "forte_options.h"
#include "mini-boost/boost/algorithm/string/predicate.hpp"
#include "operator.h"
#include "semi_canonicalize.h"
#include "psi4/libmints/dipole.h"
#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/petitelist.h"

namespace psi {
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
}

FCI_MO::FCI_MO(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), integral_(ints), mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
    print_method_banner({"Complete Active Space Configuration Interaction", "Chenyang Li"});
    startup();
}

FCI_MO::~FCI_MO() { cleanup(); }

void FCI_MO::cleanup() {}

void FCI_MO::startup() {

    // read options
    read_options();

    // setup integrals
    fci_ints_ = std::make_shared<FCIIntegrals>(integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    // compute orbital extents if CIS/CISD IPEA
    if (ipea_ != "NONE") {
        compute_orbital_extents();
    }
}

void FCI_MO::read_options() {

    // test reference type
    ref_type_ = options_.get_str("REFERENCE");
    if (ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF") {
        outfile->Printf("\n  Unrestricted reference is detected.");
        outfile->Printf("\n  We suggest using unrestricted natural orbitals.");
    }

    // active space type
    active_space_type_ = options_.get_str("FCIMO_ACTV_TYPE");

    // IP / EA
    ipea_ = options_.get_str("FCIMO_IPEA");

    // print level
    print_ = options_.get_int("PRINT");

    // energy convergence
    econv_ = options_.get_double("E_CONVERGENCE");
    fcheck_threshold_ = 100.0 * econv_;

    // nuclear repulsion
    std::shared_ptr<Molecule> molecule = Process::environment.molecule();
    e_nuc_ = molecule->nuclear_repulsion_energy();

    // digonalization algorithm
    diag_algorithm_ = options_.get_str("DIAG_ALGORITHM");

    // semicanonical orbitals
    semi_ = options_.get_bool("SEMI_CANONICAL");

    // number of Irrep
    nirrep_ = this->nirrep();

    // obtain MOs
    nmo_ = this->nmo();
    nmopi_ = this->nmopi();
    ncmo_ = mo_space_info_->size("CORRELATED");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    // obtain frozen orbitals
    frzcpi_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->get_dimension("FROZEN_UOCC");
    nfrzc_ = mo_space_info_->size("FROZEN_DOCC");
    nfrzv_ = mo_space_info_->size("FROZEN_UOCC");

    // obtain active orbitals
    if (options_["ACTIVE"].size() == 0) {
        outfile->Printf("\n  Please specify the ACTIVE occupations.");
        outfile->Printf("\n  Single-reference computations should set ACTIVE to zeros.");
        outfile->Printf("\n  For example, ACTIVE [0,0,0,0] depending on the symmetry. \n");
        throw PSIEXCEPTION("Please specify the ACTIVE occupations. Check output for details.");
    }
    active_ = mo_space_info_->get_dimension("ACTIVE");
    na_ = active_.sum();

    // obitan inactive orbitals
    core_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    virtual_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    nc_ = core_.sum();
    nv_ = virtual_.sum();

    // compute number of electrons
    int natom = molecule->natom();
    size_t nelec = 0;
    for (int i = 0; i < natom; ++i) {
        nelec += molecule->fZ(i);
    }
    int charge = molecule->molecular_charge();
    if (options_["CHARGE"].has_changed()) {
        charge = options_.get_int("CHARGE");
    }
    nelec -= charge;
    multi_ = molecule->multiplicity();
    if (options_["MULTIPLICITY"].has_changed()) {
        multi_ = options_.get_int("MULTIPLICITY");
    }
    if (multi_ < 1) {
        outfile->Printf("\n  MULTIPLICITY must be no less than 1.");
        outfile->Printf("\n  MULTIPLICITY = %2d", multi_);
        outfile->Printf("\n  Check (specify) Multiplicity! \n");
        throw PSIEXCEPTION("MULTIPLICITY must be no less than 1. Check output for details.");
    }
    twice_ms_ = std::round(2.0 * options_.get_double("MS"));
    if (twice_ms_ < 0) {
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", twice_ms_, multi_);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
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
        throw PSIEXCEPTION("Negative number of alpha electrons or beta "
                           "electrons. Check output for details.");
    }
    if (nalfa_ - nc_ - nfrzc_ > na_) {
        outfile->Printf("\n  Not enough active orbitals to arrange electrons!");
        outfile->Printf("\n  Number of orbitals: active = %5zu, core = %5zu", na_, nc_);
        outfile->Printf("\n  Number of alpha electrons: Nalpha = %5ld", nalfa_);
        outfile->Printf("\n  Check core and active orbitals! \n");
        throw PSIEXCEPTION("Not enough active orbitals to arrange electrons! "
                           "Check output for details.");
    }

    // obtain root symmetry
    root_sym_ = options_.get_int("ROOT_SYM");

    // obtain number of roots and roots of interest
    nroot_ = options_.get_int("NROOT");
    root_ = options_.get_int("ROOT");
    if (root_ >= nroot_) {
        outfile->Printf("\n  NROOT = %3d, ROOT = %3d", nroot_, root_);
        outfile->Printf("\n  ROOT must be smaller than NROOT.");
        throw PSIEXCEPTION("ROOT must be smaller than NROOT.");
    }

    // setup symmetry index of active orbitals
    for (int h = 0; h < nirrep_; ++h) {
        for (size_t i = 0; i < size_t(active_[h]); ++i) {
            sym_active_.push_back(h);
        }
    }

    // setup symmetry index of correlated orbitals
    for (int h = 0; h < nirrep_; ++h) {
        for (size_t i = 0; i < size_t(ncmopi_[h]); ++i) {
            sym_ncmo_.push_back(h);
        }
    }

    // obtain absolute indices of core, active and virtual
    idx_c_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    idx_a_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    idx_v_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // setup hole and particle indices (Active must start first for old mcsrgpt2
    // code)
    nh_ = nc_ + na_;
    npt_ = na_ + nv_;
    idx_h_ = std::vector<size_t>(idx_a_);
    idx_h_.insert(idx_h_.end(), idx_c_.begin(), idx_c_.end());
    idx_p_ = std::vector<size_t>(idx_a_);
    idx_p_.insert(idx_p_.end(), idx_v_.begin(), idx_v_.end());

    // active hole and particle indices
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        Dimension doccpi(this->doccpi());
        if (ipea_ == "EA") {
            doccpi[0] += 1;
        }
        active_h_ = doccpi - frzcpi_ - core_;
        active_p_ = active_ - active_h_;

        ah_.clear();
        ap_.clear();
        for (int h = 0; h < nirrep_; ++h) {
            int h_local = h;
            size_t offset = 0;
            while (--h_local >= 0) {
                offset += active_[h_local];
            }

            for (size_t i = 0; i < active_[h]; ++i) {
                if (i < active_h_[h]) {
                    ah_.push_back(i + offset);
                } else {
                    ap_.push_back(i + offset);
                }
            }
        }
    }

    // print input summary
    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"number of atoms", natom});
    info.push_back({"number of electrons", nelec});
    info.push_back({"molecular charge", charge});
    info.push_back({"number of alpha electrons", nalfa_});
    info.push_back({"number of beta electrons", nbeta_});
    info.push_back({"multiplicity", multi_});
    info.push_back({"ms (2 * Sz)", twice_ms_});
    info.push_back({"number of molecular orbitals", nmo_});

    if (print_ > 0) {
        print_h2("Input Summary");
    }
    if (print_ > 0) {
        for (auto& str_dim : info) {
            outfile->Printf("\n    %-30s = %5zu", str_dim.first.c_str(), str_dim.second);
        }
    }

    // print orbital spaces
    if (print_ > 0) {
        print_h2("Orbital Spaces");
        print_irrep("TOTAL MO", nmopi_);
        print_irrep("FROZEN CORE", frzcpi_);
        print_irrep("FROZEN VIRTUAL", frzvpi_);
        print_irrep("CORRELATED MO", ncmopi_);
        print_irrep("CORE", core_);
        print_irrep("ACTIVE", active_);
        print_irrep("VIRTUAL", virtual_);
    }

    // state averaging
    if (options_["AVG_STATE"].size() != 0) {

        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
        std::vector<std::string> irrep_symbol;
        for (int h = 0; h < nirrep_; ++h) {
            irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
        }

        size_t nstates = 0;
        int nentry = options_["AVG_STATE"].size();

        // figure out total number of states
        std::vector<int> nstatespim;
        std::vector<int> irreps;
        std::vector<int> multis;
        for (int i = 0; i < nentry; ++i) {
            if (options_["AVG_STATE"][i].size() != 3) {
                outfile->Printf("\n  Error: invalid input of AVG_STATE. Each "
                                "entry should take an array of three numbers.");
                throw PSIEXCEPTION("Invalid input of AVG_STATE");
            }

            // irrep
            int irrep = options_["AVG_STATE"][i][0].to_integer();
            if (irrep >= nirrep_ || irrep < 0) {
                outfile->Printf("\n  Error: invalid irrep in AVG_STATE. Please "
                                "check the input irrep (start from 0) not to "
                                "exceed %d",
                                nirrep_ - 1);
                throw PSIEXCEPTION("Invalid irrep in AVG_STATE");
            }
            irreps.push_back(irrep);

            // multiplicity
            int multi = options_["AVG_STATE"][i][1].to_integer();
            if (multi < 1) {
                outfile->Printf("\n  Error: invalid multiplicity in AVG_STATE.");
                throw PSIEXCEPTION("Invaid multiplicity in AVG_STATE");
            }
            multis.push_back(multi);

            // number of states of irrep and multiplicity
            int nstates_this = options_["AVG_STATE"][i][2].to_integer();
            if (nstates_this < 1) {
                outfile->Printf("\n  Error: invalid nstates in AVG_STATE. "
                                "nstates of a certain irrep and multiplicity "
                                "should greater than 0.");
                throw PSIEXCEPTION("Invalid nstates in AVG_STATE.");
            }
            nstatespim.push_back(nstates_this);
            nstates += nstates_this;
        }

        // test input weights
        std::vector<std::vector<double>> weights;
        if (options_["AVG_WEIGHT"].has_changed()) {
            if (options_["AVG_WEIGHT"].size() != nentry) {
                outfile->Printf("\n  Error: mismatched number of entries in "
                                "AVG_STATE (%d) and AVG_WEIGHT (%d).",
                                nentry, options_["AVG_WEIGHT"].size());
                throw PSIEXCEPTION("Mismatched number of entries in AVG_STATE "
                                   "and AVG_WEIGHT.");
            }

            double wsum = 0.0;
            for (int i = 0; i < nentry; ++i) {
                int nw = options_["AVG_WEIGHT"][i].size();
                if (nw != nstatespim[i]) {
                    outfile->Printf("\n  Error: mismatched number of weights "
                                    "in entry %d of AVG_WEIGHT. Asked for %d "
                                    "states but only %d weights.",
                                    i, nstatespim[i], nw);
                    throw PSIEXCEPTION("Mismatched number of weights in AVG_WEIGHT.");
                }

                std::vector<double> weight;
                for (int n = 0; n < nw; ++n) {
                    double w = options_["AVG_WEIGHT"][i][n].to_double();
                    if (w < 0.0) {
                        outfile->Printf("\n  Error: negative weights in AVG_WEIGHT.");
                        throw PSIEXCEPTION("Negative weights in AVG_WEIGHT.");
                    }
                    weight.push_back(w);
                    wsum += w;
                }
                weights.push_back(weight);
            }
            if (std::fabs(wsum - 1.0) > 1.0e-10) {
                outfile->Printf("\n  Error: AVG_WEIGHT entries do not add up "
                                "to 1.0. Sum = %.10f",
                                wsum);
                throw PSIEXCEPTION("AVG_WEIGHT entries do not add up to 1.0.");
            }

        } else {
            // use equal weights
            double w = 1.0 / nstates;
            for (int i = 0; i < nentry; ++i) {
                std::vector<double> weight(nstatespim[i], w);
                weights.push_back(weight);
            }
        }

        // form option parser
        for (int i = 0; i < nentry; ++i) {
            std::tuple<int, int, int, std::vector<double>> avg_info =
                std::make_tuple(irreps[i], multis[i], nstatespim[i], weights[i]);
            sa_info_.push_back(avg_info);
        }

        // printing summary
        print_h2("State Averaging Summary");
        int lweight = *std::max_element(nstatespim.begin(), nstatespim.end());
        if (lweight == 1) {
            lweight = 7;
        } else {
            lweight *= 6;
            lweight -= 1;
        }
        int ltotal = 6 + 2 + 6 + 2 + 7 + 2 + lweight;
        std::string blank(lweight - 7, ' ');
        std::string dash(ltotal, '-');
        outfile->Printf("\n    Irrep.  Multi.  Nstates  %sWeights", blank.c_str());
        outfile->Printf("\n    %s", dash.c_str());
        for (int i = 0; i < nentry; ++i) {
            std::string w_str;
            for (double w : weights[i]) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << w;
                w_str += ss.str() + " ";
            }
            w_str.pop_back(); // delete the last space character

            std::stringstream ss;
            ss << std::setw(4) << std::right << irrep_symbol[irreps[i]] << "    " << std::setw(4)
               << std::right << multis[i] << "    " << std::setw(5) << std::right << nstatespim[i]
               << "    " << std::setw(lweight) << w_str;
            outfile->Printf("\n    %s", ss.str().c_str());
        }
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n    Total number of states: %d", nstates);
        outfile->Printf("\n    %s\n", dash.c_str());
    }
}

double FCI_MO::compute_energy() {
    if (options_["AVG_STATE"].size() != 0) {
        Eref_ = compute_sa_energy();
    } else {
        Eref_ = compute_ss_energy();
    }

    Process::environment.globals["CURRENT ENERGY"] = Eref_;
    return Eref_;
}

double FCI_MO::compute_ss_energy() {
    // form determinants
    form_p_space();

    // diagonalize the CASCI Hamiltonian
    bool noHF = options_.get_bool("FCIMO_CISD_NOHF");
    if (multi_ == 1 && root_sym_ == 0 &&
        (active_space_type_ == "CIS" || (active_space_type_ == "CISD" && noHF))) {
        Diagonalize_H_noHF(determinant_, multi_, nroot_, eigen_);
    } else {
        Diagonalize_H(determinant_, multi_, nroot_, eigen_);
    }

    // print CI vectors in eigen_
    size_t eigen_size = eigen_.size();
    if (nroot_ > eigen_size) {
        outfile->Printf("\n  Too many roots of interest!");
        std::string be = (eigen_size > 1) ? "are" : "is";
        std::string plural = (eigen_size > 1) ? "roots" : "root";
        outfile->Printf("\n  There %s only %3d %s that satisfy the condition!", be.c_str(),
                        eigen_size, plural.c_str());
        outfile->Printf("\n  Check root_sym, multi, and number of determinants.");
        throw PSIEXCEPTION("Too many roots of interest.");
    }
    print_CI(nroot_, options_.get_double("FCIMO_PRINT_CIVEC"), eigen_, determinant_);

    // compute dipole moments
    compute_permanent_dipole();

    // compute oscillator strength
    if (nroot_ > 1) {
        compute_transition_dipole();
        compute_oscillator_strength();
    }

    double Eref = eigen_[root_].second;
    Eref_ = Eref;
    Process::environment.globals["CURRENT ENERGY"] = Eref;
    return Eref;
}

void FCI_MO::form_p_space() {
    // clean previous determinants
    determinant_.clear();

    // form determinants
    if (active_space_type_ == "CIS") {
        form_det_cis();
    } else if (active_space_type_ == "CISD") {
        form_det_cisd();
    } else {
        form_det();
    }
}

void FCI_MO::form_det() {

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // Alpha and Beta Strings
    Timer tstrings;
    std::string str = "Forming alpha and beta strings";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }
    std::vector<vector<vector<bool>>> a_string = Form_String(na_a);
    std::vector<vector<vector<bool>>> b_string = Form_String(nb_a);
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tstrings.get());
    }

    // Form Determinant
    Timer tdet;
    str = "Forming determinants";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }
    if (active_space_type_ == "DOCI") {
        if (root_sym_ != 0 || multi_ != 1) {
            outfile->Printf("\n  State must be totally symmetric for DOCI.");
            throw PSIEXCEPTION("State must be totally symmetric for DOCI.");
        } else {
            for (int i = 0; i != nirrep_; ++i) {
                size_t sa = a_string[i].size();
                for (size_t alfa = 0; alfa < sa; ++alfa) {
                    determinant_.push_back(
                        STLBitsetDeterminant(a_string[i][alfa], a_string[i][alfa]));
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
                    determinant_.push_back(
                        STLBitsetDeterminant(a_string[i][alfa], b_string[j][beta]));
                }
            }
        }
    }
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }

    // printing
    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    info.push_back({"number of determinants", determinant_.size()});

    print_h2("Determinants Summary");
    for (auto& str_dim : info) {
        outfile->Printf("\n    %-35s = %5zu", str_dim.first.c_str(), str_dim.second);
    }
    outfile->Printf("\n");

    if (print_ > 1) {
        print_det(determinant_);
    }

    if (determinant_.size() == 0) {
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Check the wavefunction symmetry and multiplicity.");
        throw PSIEXCEPTION("No determinant matching the conditions!");
    }
}

vector<vector<vector<bool>>> FCI_MO::Form_String(const int& active_elec, const bool& print) {

    timer_on("FORM String");
    std::vector<vector<vector<bool>>> String(nirrep_, std::vector<vector<bool>>());

    // initalize the string (only active)
    int symmetry = 0;
    bool* I_init = new bool[na_];
    for (size_t i = 0; i < na_; ++i)
        I_init[i] = 0;
    for (size_t i = na_ - active_elec; i < na_; ++i)
        I_init[i] = 1;

    do {
        // permute the active
        std::vector<bool> string_a;
        int sym = symmetry;
        for (size_t i = 0; i < na_; ++i) {
            string_a.push_back(I_init[i]);
            if (I_init[i] == 1) {
                sym ^= sym_active_[i];
            }
        }
        String[sym].push_back(string_a);
    } while (std::next_permutation(I_init, I_init + na_));

    if (print == true && !quiet_) {
        print_h2("Possible String");
        for (size_t i = 0; i != String.size(); ++i) {
            outfile->Printf("\n  symmetry = %lu \n", i);
            for (size_t j = 0; j != String[i].size(); ++j) {
                outfile->Printf("    ");
                for (bool b : String[i][j]) {
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
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
    for (int i = 0; i < na_; ++i) {
        if (string_ref[i]) {
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    Timer tdet;
    string str = "Forming determinants";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }

    int i = symmetry ^ root_sym_;
    size_t single_size = string_singles[i].size();
    for (size_t x = 0; x < single_size; ++x) {
        determinant_.push_back(STLBitsetDeterminant(string_singles[i][x], string_ref));
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles[i][x]));
    }

    // add HF determinant at the end if root_sym = 0
    if (root_sym_ == 0) {
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_ref));
    }

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // printing
    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    if (root_sym_ == 0) {
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
        throw PSIEXCEPTION("No determinant matching the conditions!");
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
    for (int i = 0; i < na_; ++i) {
        if (string_ref[i]) {
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    Timer tdet;
    string str = "Forming determinants";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }

    int i = symmetry ^ root_sym_;
    singles_size_ = 0;
    if (ipea_ == "NONE") {
        size_t single_size = string_singles[i].size();
        singles_size_ = 2 * single_size;
        for (size_t x = 0; x < single_size; ++x) {
            determinant_.push_back(STLBitsetDeterminant(string_singles[i][x], string_ref));
            determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles[i][x]));
        }
    } else {
        size_t single_size = string_singles_ipea[i].size();
        singles_size_ = 2 * single_size;
        for (size_t x = 0; x < single_size; ++x) {
            determinant_.push_back(STLBitsetDeterminant(string_singles_ipea[i][x], string_ref));
            determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles_ipea[i][x]));
        }
    }

    // doubles
    size_t double_size = string_doubles[i].size();
    for (size_t x = 0; x < double_size; ++x) {
        determinant_.push_back(STLBitsetDeterminant(string_doubles[i][x], string_ref));
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_doubles[i][x]));
    }

    for (int h = 0; h < nirrep_; ++h) {
        size_t single_size_a = string_singles[h].size();
        for (size_t x = 0; x < single_size_a; ++x) {
            int sym = h ^ root_sym_;

            size_t single_size_b = string_singles[sym].size();
            for (size_t y = 0; y < single_size_b; ++y) {
                determinant_.push_back(
                    STLBitsetDeterminant(string_singles[h][x], string_singles[sym][y]));
            }

            if (ipea_ != "NONE") {
                size_t single_ipea_size_b = string_singles_ipea[sym].size();
                for (size_t y = 0; y < single_ipea_size_b; ++y) {
                    determinant_.push_back(
                        STLBitsetDeterminant(string_singles[h][x], string_singles_ipea[sym][y]));
                    determinant_.push_back(
                        STLBitsetDeterminant(string_singles_ipea[sym][y], string_singles[h][x]));
                }
            }
        }
    }

    // add HF determinant at the end if root_sym = 0
    if (root_sym_ == 0) {
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_ref));
    }

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdet.get());
    }

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // printing
    std::vector<std::pair<std::string, size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    if (root_sym_ == 0) {
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
        throw PSIEXCEPTION("No determinant matching the conditions!");
    }
}

vector<bool> FCI_MO::Form_String_Ref(const bool& print) {
    timer_on("FORM String Ref");

    std::vector<bool> String;
    for (int h = 0; h < nirrep_; ++h) {
        int act_docc = active_h_[h];
        int act = active_[h];
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
    for (size_t i = 0; i < na_; ++i) {
        if (ipea_ != "NONE" &&
            std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()) {
            continue;
        }

        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        } else {
            uocc.push_back(i);
        }
    }

    // singles
    for (const int& a : uocc) {
        std::vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_active_[a];
        for (const int& i : occ) {
            string_local[i] = false;
            sym ^= sym_active_[i];
            String[sym].push_back(string_local);
            // need to reset
            string_local[i] = true;
            sym ^= sym_active_[i];
        }
    }

    if (print) {
        print_h2("Singles String");
        for (size_t i = 0; i != String.size(); ++i) {
            if (String[i].size() != 0) {
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for (size_t j = 0; j != String[i].size(); ++j) {
                outfile->Printf("    ");
                for (bool b : String[i][j]) {
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
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
    for (int i = 0; i < na_; ++i) {
        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    for (const int& i : occ) {
        std::vector<bool> string_local(ref_string);
        string_local[idx_diffused_] = true;

        string_local[i] = false;
        int sym = symmetry ^ sym_active_[i];
        String[sym].push_back(string_local);
    }

    if (print) {
        print_h2("Singles String IP");
        for (size_t i = 0; i != String.size(); ++i) {
            if (String[i].size() != 0) {
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for (size_t j = 0; j != String[i].size(); ++j) {
                outfile->Printf("    ");
                for (bool b : String[i][j]) {
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
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
    for (int i = 0; i < na_; ++i) {
        if (!ref_string[i]) {
            uocc.push_back(i);
        } else {
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    for (const int& a : uocc) {
        std::vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_active_[a];

        string_local[idx_diffused_] = false;
        String[sym].push_back(string_local);
    }

    if (print) {
        print_h2("Singles String EA");
        for (size_t i = 0; i != String.size(); ++i) {
            if (String[i].size() != 0) {
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for (size_t j = 0; j != String[i].size(); ++j) {
                outfile->Printf("    ");
                for (bool b : String[i][j]) {
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
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
    for (int i = 0; i < na_; ++i) {
        if (ipea_ != "NONE" && i != idx_diffused_ &&
            std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()) {
            continue;
        }

        if (ref_string[i]) {
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        } else {
            uocc.push_back(i);
        }
    }

    // doubles
    for (const int& a : uocc) {
        std::vector<bool> string_a(ref_string);
        string_a[a] = true;
        int sym_a = symmetry ^ sym_active_[a];

        for (const int& b : uocc) {
            if (b > a) {
                std::vector<bool> string_b(string_a);
                string_b[b] = true;
                int sym_b = sym_a ^ sym_active_[b];

                for (const int& i : occ) {
                    std::vector<bool> string_i(string_b);
                    string_i[i] = false;
                    int sym_i = sym_b ^ sym_active_[i];

                    for (const int& j : occ) {
                        if (j > i) {
                            std::vector<bool> string_j(string_i);
                            string_j[j] = false;
                            int sym_j = sym_i ^ sym_active_[j];
                            String[sym_j].push_back(string_j);
                        }
                    }
                }
            }
        }
    }

    if (print) {
        print_h2("Doubles String");
        for (size_t i = 0; i != String.size(); ++i) {
            if (String[i].size() != 0) {
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for (size_t j = 0; j != String[i].size(); ++j) {
                outfile->Printf("    ");
                for (bool b : String[i][j]) {
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    timer_off("FORM String Doubles");
    return String;
}

vector<double> FCI_MO::compute_T1_percentage() {
    std::vector<double> out;

    if (active_space_type_ != "CISD") {
        outfile->Printf("\n  No point to compute T1 percentage. Return an empty vector.");
    } else {
        // in consistent to form_det_cisd,
        // the first singles_size_ determinants in determinant_ are singles
        for (size_t n = 0, eigen_size = eigen_.size(); n < eigen_size; ++n) {
            double t1 = 0;
            SharedVector evec = eigen_[n].first;
            for (size_t i = 0; i < singles_size_; ++i) {
                double v = evec->get(i);
                t1 += v * v;
            }
            out.push_back(100.0 * t1);
        }
    }

    return out;
}

void FCI_MO::semi_canonicalize() {
    SharedMatrix Ua(new Matrix("Unitary A", nmopi_, nmopi_));
    SharedMatrix Ub(new Matrix("Unitary B", nmopi_, nmopi_));
    BD_Fock(Fa_, Fb_, Ua, Ub, "Fock");
    SharedMatrix Ca = this->Ca();
    SharedMatrix Cb = this->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false, false, 1.0, Ca, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, Ub, 0.0);

    // overlap of original and semicanonical orbitals
    SharedMatrix MOoverlap = Matrix::triplet(Ca, this->S(), Ca_new, true, false, false);
    MOoverlap->set_name("MO overlap");

    // copy semicanonical orbital to wavefunction
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // test active orbital ordering
    for (int h = 0; h < nirrep_; ++h) {
        int actv_start = frzcpi_[h] + core_[h];
        int actv_end = actv_start + active_[h];

        std::map<int, int> indexmap;
        std::vector<int> idx_0;
        for (int i = actv_start; i < actv_end; ++i) {
            int ii = 0; // corresponding index in semicanonical basis
            double smax = 0.0;

            for (int j = actv_start; j < actv_end; ++j) {
                double s = MOoverlap->get(h, i, j);
                if (std::fabs(s) > smax) {
                    smax = std::fabs(s);
                    ii = j;
                }
            }

            if (ii != i) {
                indexmap[i] = ii;
                idx_0.push_back(i);
            }
        }

        // find orbitals to swap if the loop is closed
        std::vector<int> idx_swap;
        for (const int& x : idx_0) {
            // if index x is already in the to-be-swapped index, then continue
            if (std::find(idx_swap.begin(), idx_swap.end(), x) != idx_swap.end()) {
                continue;
            }

            std::vector<int> temp;
            int local = x;

            while (indexmap.find(indexmap[local]) != indexmap.end()) {
                if (std::find(temp.begin(), temp.end(), local) == temp.end()) {
                    temp.push_back(local);
                } else {
                    // a loop found
                    break;
                }

                local = indexmap[local];
            }

            // start from the point that has the value of "local" and copy to
            // idx_swap
            int pos = std::find(temp.begin(), temp.end(), local) - temp.begin();
            for (int i = pos; i < temp.size(); ++i) {
                if (std::find(idx_swap.begin(), idx_swap.end(), temp[i]) == idx_swap.end()) {
                    idx_swap.push_back(temp[i]);
                }
            }
        }

        // remove the swapped orbitals from the vector of orginal orbitals
        idx_0.erase(std::remove_if(idx_0.begin(), idx_0.end(),
                                   [&](int i) {
                                       return std::find(idx_swap.begin(), idx_swap.end(), i) !=
                                              idx_swap.end();
                                   }),
                    idx_0.end());

        // swap orbitals
        for (const int& x : idx_swap) {
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while ((--h_local) >= 0) {
                ni += ncmopi_[h_local];
                nj += ncmopi_[h_local];
            }
            outfile->Printf("\n  Orbital ordering changed due to "
                            "semicanonicalization. Swapped orbital %3zu back "
                            "to %3zu.",
                            nj, ni);
            Ca->set_column(h, x, Ca_new->get_column(h, indexmap[x]));
            Cb->set_column(h, x, Cb_new->get_column(h, indexmap[x]));
        }

        // throw warnings when inconsistency is detected
        for (const int& x : idx_0) {
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while ((--h_local) >= 0) {
                ni += ncmopi_[h_local];
                nj += ncmopi_[h_local];
            }
            outfile->Printf("\n  Orbital %3zu may have changed to "
                            "semicanonical orbital %3zu. Please interpret "
                            "orbitals with caution.",
                            ni, nj);
        }
    }

    outfile->Printf("\n\n");
    integral_->retransform_integrals();
    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();
}

void FCI_MO::Diagonalize_H_noHF(const vecdet& p_space, const int& multi, const int& nroot,
                                std::vector<pair<SharedVector, double>>& eigen) {
    // recompute RHF determinant
    std::vector<bool> string_ref = Form_String_Ref();
    STLBitsetDeterminant rhf(string_ref, string_ref);

    // test if RHF determinant is the last one in det
    STLBitsetDeterminant det_back = p_space.back();
    if (rhf == det_back) {
        eigen.clear();
        size_t det_size = p_space.size();

        // compute RHF energy
        outfile->Printf("\n  Isolate RHF determinant to the rest determinants.");
        outfile->Printf("\n  Recompute RHF energy ... ");
        double Erhf = fci_ints_->energy(rhf) + fci_ints_->scalar_energy() + e_nuc_;
        SharedVector rhf_vec(new Vector("RHF Eigen Vector", det_size));
        rhf_vec->set(det_size - 1, 1.0);
        eigen.push_back(std::make_pair(rhf_vec, Erhf));
        outfile->Printf("Done.");

        // compute the rest of the states
        if (nroot > 1) {
            outfile->Printf("\n  The upcoming diagonalization excludes RHF determinant.\n");

            int nroot_noHF = nroot - 1;
            vecdet p_space_noHF(p_space);
            p_space_noHF.pop_back();
            std::vector<pair<SharedVector, double>> eigen_noHF;
            Diagonalize_H(p_space_noHF, multi, nroot_noHF, eigen_noHF);

            for (int i = 0; i < nroot_noHF; ++i) {
                SharedVector vec_noHF = eigen_noHF[i].first;
                double Ethis = eigen_noHF[i].second;

                string name = "Root " + std::to_string(i) + " Eigen Vector";
                SharedVector vec(new Vector(name, det_size));
                for (size_t n = 0; n < det_size - 1; ++n) {
                    vec->set(n, vec_noHF->get(n));
                }

                eigen.push_back(std::make_pair(vec, Ethis));
            }
        }

    } else {
        outfile->Printf("\n  Error: RHF determinant NOT at the end of the "
                        "determinant vector.");
        outfile->Printf("\n    Diagonalize_H_noHF only works for root_sym = 0.");
        throw PSIEXCEPTION("RHF determinant NOT at the end of determinant "
                           "vector. Problem at Diagonalize_H_noHF of FCI_MO.");
    }
}

void FCI_MO::Diagonalize_H(const vecdet& p_space, const int& multi, const int& nroot,
                           std::vector<pair<SharedVector, double>>& eigen) {
    timer_on("Diagonalize H");
    Timer tdiagH;
    std::string str = "Diagonalizing Hamiltonian";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }
    size_t det_size = p_space.size();
    eigen.clear();

    //    // use bitset determinants
    //    STLBitsetDeterminant::set_ints(fci_ints_);
    //    std::vector<STLBitsetDeterminant> P_space;
    //    for (size_t x = 0; x != det_size; ++x) {
    //        std::vector<bool> alfa_bits =
    //        P_space[x].get_alfa_bits_vector_bool();
    //        std::vector<bool> beta_bits =
    //        P_space[x].get_beta_bits_vector_bool();
    //        STLBitsetDeterminant bs_det(alfa_bits, beta_bits);
    //        P_space.push_back(bs_det);
    //        //        bs_det.print();
    //    }

    // DL solver
    SparseCISolver sparse_solver(fci_ints_);
    DiagonalizationMethod diag_method = DLSolver;
    string sigma_method = options_.get_str("SIGMA_BUILD_TYPE");
    sparse_solver.set_e_convergence(econv_);
    sparse_solver.set_spin_project(true);
    sparse_solver.set_maxiter_davidson(options_.get_int("DL_MAXITER"));
    sparse_solver.set_guess_dimension(options_.get_int("DL_GUESS_SIZE"));
    sparse_solver.set_sigma_method(sigma_method);
    if (!quiet_) {
        sparse_solver.set_print_details(true);
    }

    // setup eigen values and vectors
    SharedMatrix evecs;
    SharedVector evals;

    // diagnoalize the Hamiltonian
    if (det_size <= 200) {
        // full Hamiltonian if detsize <= 200
        diag_method = Full;
        sparse_solver.diagonalize_hamiltonian(p_space, evals, evecs, nroot, multi, diag_method);

        // fill in eigen
        double energy_offset = fci_ints_->scalar_energy() + e_nuc_;
        for (int i = 0; i != nroot; ++i) {
            double value = evals->get(i);

            eigen.push_back(std::make_pair(evecs->get_column(0, i), value + energy_offset));
        }

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

        // add doubly occupied energy and nuclear repulsion
        // fill in eigen (no need to test spin)
        double energy_offset = fci_ints_->scalar_energy() + e_nuc_;
        for (int i = 0; i != nroot; ++i) {
            double value = evals->get(i);

            eigen.push_back(std::make_pair(evecs->get_column(0, i), value + energy_offset));
        }
    }

    //    // check spin
    //    int count = 0;
    //    if (!quiet_) {
    //        outfile->Printf("\n\n  Reference type: %s", ref_type_.c_str());
    //    }
    //    double threshold = 0.1;
    //    if (ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF") {
    //        threshold =
    //            0.20 *
    //            multi_; // 20% off from the multiplicity of the spin eigen
    //            state
    //    }
    //    if (!quiet_) {
    //        outfile->Printf("\n  Threshold for spin check: %.2f", threshold);
    //    }

    //    for (int i = 0; i != nroot; ++i) {
    //        double S2 = 0.0;
    //        outfile->Printf("\n  1551");
    //        for (int I = 0; I < det_size; ++I) {
    //            outfile->Printf("\n  1553");
    //            for (int J = 0; J < det_size; ++J) {
    //                double S2IJ = P_space[I].spin2(P_space[J]);
    //                S2 += S2IJ * vec_tmp->get(I, i) * vec_tmp->get(J, i);
    //            }
    //        }
    //        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
    //        double multi_real = 2.0 * S + 1;

    //        if (std::fabs(multi_ - multi_real) > threshold) {
    //            if (!quiet_) {
    //                outfile->Printf("\n\n  Ask for S^2 = %.4f, this S^2 =
    //                %.4f, "
    //                                "continue searching...",
    //                                0.25 * (multi_ * multi_ - 1.0), S2);
    //            }
    //            continue;
    //        } else {
    //            std::vector<std::string> s2_labels(
    //                {"singlet", "doublet", "triplet", "quartet", "quintet",
    //                 "sextet", "septet", "octet", "nonet", "decaet"});
    //            std::string state_label = s2_labels[std::round(S * 2.0)];
    //            if (!quiet_) {
    //                outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f,
    //                %s "
    //                                "(from %zu determinants)",
    //                                S2, S, state_label.c_str(), det_size);
    //            }
    //            ++count;
    //            eigen.push_back(
    //                std::make_pair(vec_tmp->get_column(0, i), val_tmp->get(i) +
    //                e_nuc_));
    //        }
    //        if (count == nroot)
    //            break;
    //    }
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdiagH.get());
    }
    timer_off("Diagonalize H");
}

inline bool ReverseAbsSort(const tuple<double, int>& lhs, const tuple<double, int>& rhs) {
    return std::fabs(std::get<0>(rhs)) < std::fabs(std::get<0>(lhs));
}

void FCI_MO::print_CI(const int& nroot, const double& CI_threshold,
                      const std::vector<pair<SharedVector, double>>& eigen, const vecdet& det) {
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
        sort(ci_select.begin(), ci_select.end(), ReverseAbsSort);
        dominant_dets_.push_back(det[std::get<1>(ci_select[0])]);

        if (!quiet_) {
            outfile->Printf("\n  ==> Root No. %d <==\n", i);
        }
        for (size_t j = 0, ci_select_size = ci_select.size(); j < ci_select_size; ++j) {
            if (!quiet_) {
                outfile->Printf("\n    ");
            }
            double ci = std::get<0>(ci_select[j]);
            size_t index = std::get<1>(ci_select[j]);
            size_t ncmopi = 0;
            for (int h = 0; h < nirrep_; ++h) {
                for (size_t k = 0; k < active_[h]; ++k) {
                    size_t x = k + ncmopi;
                    bool a = det[index].get_alfa_bit(x);
                    bool b = det[index].get_beta_bit(x);
                    if (a == b) {
                        if (!quiet_) {
                            outfile->Printf("%d", a == 1 ? 2 : 0);
                        }
                    } else {
                        if (!quiet_) {
                            outfile->Printf("%c", a == 1 ? 'a' : 'b');
                        }
                    }
                }
                if (active_[h] != 0)
                    if (!quiet_) {
                        outfile->Printf(" ");
                    }
                ncmopi += active_[h];
            }
            if (!quiet_) {
                outfile->Printf(" %20.10f", ci);
            }
        }
        if (!quiet_) {
            outfile->Printf("\n\n    Total Energy:   %.15lf\n\n", eigen[i].second);
        }
    }

    timer_off("Print CI Vectors");
}

void FCI_MO::FormDensity(CI_RDMS& ci_rdms, d2& A, d2& B) {
    timer_on("FORM Density");
    Timer tdensity;
    std::string str = "Forming one-particle density";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }

    for (size_t p = 0; p < nc_; ++p) {
        size_t np = idx_c_[p];
        A[np][np] = 1.0;
        B[np][np] = 1.0;
    }

    size_t dim = na_ * na_;
    std::vector<double> opdm_a(dim, 0.0);
    std::vector<double> opdm_b(dim, 0.0);

    ci_rdms.compute_1rdm(opdm_a, opdm_b);

    for (size_t p = 0; p < na_; ++p) {
        size_t np = idx_a_[p];
        for (size_t q = p; q < na_; ++q) {
            size_t nq = idx_a_[q];

            if ((sym_active_[p] ^ sym_active_[q]) != 0)
                continue;

            size_t index = p * na_ + q;
            A[np][nq] = opdm_a[index];
            B[np][nq] = opdm_b[index];

            A[nq][np] = A[np][nq];
            B[nq][np] = B[np][nq];
        }
    }

    fill_density();
    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tdensity.get());
    }
    timer_off("FORM Density");
}

// double FCI_MO::OneOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t& p,
//                     const bool& sp, const size_t& q, const bool& sq) {
//    timer_on("1PO");
//    std::vector<vector<bool>> tmp;
//    tmp.push_back(J.get_alfa_bits_vector_bool());
//    tmp.push_back(J.get_beta_bits_vector_bool());

//    double sign = 1.0;

//    if (tmp[sq][q]) {
//        sign *= CheckSign(tmp[sq], q);
//        tmp[sq][q] = 0;
//    } else {
//        timer_off("1PO");
//        return 0.0;
//    }

//    if (!tmp[sp][p]) {
//        sign *= CheckSign(tmp[sp], p);
//        tmp[sp][p] = 1;
//        Jnew = STLBitsetDeterminant(tmp[0], tmp[1]);
//        timer_off("1PO");
//        return sign;
//    } else {
//        timer_off("1PO");
//        return 0.0;
//    }
//}

void FCI_MO::print_density(const string& spin, const d2& density) {
    string name = "Density " + spin;
    outfile->Printf("  ==> %s <==\n\n", name.c_str());

    SharedMatrix dens(new Matrix("A-A", na_, na_));
    for (size_t u = 0; u < na_; ++u) {
        size_t nu = idx_a_[u];
        for (size_t v = 0; v < na_; ++v) {
            size_t nv = idx_a_[v];
            dens->set(u, v, density[nu][nv]);
        }
    }

    dens->print();
}

// void FCI_MO::print_d2(const string& str, const d2& OnePD) {
//    timer_on("PRINT Density");
//    SharedMatrix M(new Matrix(str.c_str(), OnePD.size(), OnePD[0].size()));
//    for (size_t i = 0; i != OnePD.size(); ++i) {
//        for (size_t j = 0; j != OnePD[i].size(); ++j) {
//            M->pointer()[i][j] = OnePD[i][j];
//        }
//    }
//    M->print();
//    timer_off("PRINT Density");
//}

// void FCI_MO::FormCumulant2(CI_RDMS& ci_rdms, d4& AA, d4& AB, d4& BB) {
//    timer_on("FORM 2-Cumulant");
//    Timer tL2;
//    std::string str = "Forming Lambda2";
//    if (!quiet_) {
//        outfile->Printf("\n  %-35s ...", str.c_str());
//    }

//    size_t dim = na_ * na_ * na_ * na_;
//    std::vector<double> tpdm_aa(dim, 0.0);
//    std::vector<double> tpdm_ab(dim, 0.0);
//    std::vector<double> tpdm_bb(dim, 0.0);

//    ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

//    FormCumulant2AA(tpdm_aa, tpdm_bb, AA, BB);
//    FormCumulant2AB(tpdm_ab, AB);
//    //    fill_cumulant2();

//    outfile->Printf("  Done. Timing %15.6f s", tL2.get());
//    timer_off("FORM 2-Cumulant");
//}

// void FCI_MO::FormCumulant2AA(const std::vector<double>& tpdm_aa, const std::vector<double>&
// tpdm_bb,
//                             d4& AA, d4& BB) {
//    size_t dim2 = na_ * na_;
//    size_t dim3 = na_ * dim2;

//    for (size_t p = 0; p < na_; ++p) {
//        size_t np = idx_a_[p];
//        for (size_t q = p + 1; q < na_; ++q) {
//            size_t nq = idx_a_[q];
//            for (size_t r = 0; r < na_; ++r) {
//                size_t nr = idx_a_[r];
//                for (size_t s = r + 1; s < na_; ++s) {
//                    size_t ns = idx_a_[s];

//                    if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0)
//                        continue;

//                    size_t index = p * dim3 + q * dim2 + r * na_ + s;

//                    AA[p][q][r][s] += tpdm_aa[index];
//                    BB[p][q][r][s] += tpdm_bb[index];

//                    AA[p][q][r][s] -= Da_[np][nr] * Da_[nq][ns];
//                    AA[p][q][r][s] += Da_[np][ns] * Da_[nq][nr];
//                    AA[p][q][s][r] -= AA[p][q][r][s];
//                    AA[q][p][r][s] -= AA[p][q][r][s];
//                    AA[q][p][s][r] += AA[p][q][r][s];

//                    BB[p][q][r][s] -= Db_[np][nr] * Db_[nq][ns];
//                    BB[p][q][r][s] += Db_[np][ns] * Db_[nq][nr];
//                    BB[p][q][s][r] -= BB[p][q][r][s];
//                    BB[q][p][r][s] -= BB[p][q][r][s];
//                    BB[q][p][s][r] += BB[p][q][r][s];
//                }
//            }
//        }
//    }
//}

// void FCI_MO::FormCumulant2AB(const std::vector<double>& tpdm_ab, d4& AB) {
//    size_t dim2 = na_ * na_;
//    size_t dim3 = na_ * dim2;

//    for (size_t p = 0; p < na_; ++p) {
//        size_t np = idx_a_[p];
//        for (size_t q = 0; q < na_; ++q) {
//            size_t nq = idx_a_[q];
//            for (size_t r = 0; r < na_; ++r) {
//                size_t nr = idx_a_[r];
//                for (size_t s = 0; s < na_; ++s) {
//                    size_t ns = idx_a_[s];

//                    if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0)
//                        continue;

//                    size_t index = p * dim3 + q * dim2 + r * na_ + s;
//                    AB[p][q][r][s] += tpdm_ab[index];

//                    AB[p][q][r][s] -= Da_[np][nr] * Db_[nq][ns];
//                }
//            }
//        }
//    }
//}

void FCI_MO::print2PDC(const string& str, const d4& TwoPDC, const int& PRINT) {
    timer_on("PRINT 2-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    size_t size = TwoPDC.size();
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != size; ++j) {
            for (size_t k = 0; k != size; ++k) {
                for (size_t l = 0; l != size; ++l) {
                    if (std::fabs(TwoPDC[i][j][k][l]) > 1.0e-15) {
                        ++count;
                        if (PRINT > 2)
                            outfile->Printf("\n  Lambda "
                                            "[%3lu][%3lu][%3lu][%3lu] = "
                                            "%18.15lf",
                                            i, j, k, l, TwoPDC[i][j][k][l]);
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 2-Cumulant");
}

// double FCI_MO::TwoOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t& p,
//                     const bool& sp, const size_t& q, const bool& sq, const size_t& r,
//                     const bool& sr, const size_t& s, const bool& ss) {
//    timer_on("2PO");
//    std::vector<vector<bool>> tmp;
//    tmp.push_back(J.get_alfa_bits_vector_bool());
//    tmp.push_back(J.get_beta_bits_vector_bool());

//    double sign = 1.0;

//    if (tmp[sr][r]) {
//        sign *= CheckSign(tmp[sr], r);
//        tmp[sr][r] = 0;
//    } else {
//        timer_off("2PO");
//        return 0.0;
//    }

//    if (tmp[ss][s]) {
//        sign *= CheckSign(tmp[ss], s);
//        tmp[ss][s] = 0;
//    } else {
//        timer_off("2PO");
//        return 0.0;
//    }

//    if (!tmp[sq][q]) {
//        sign *= CheckSign(tmp[sq], q);
//        tmp[sq][q] = 1;
//    } else {
//        timer_off("2PO");
//        return 0.0;
//    }

//    if (!tmp[sp][p]) {
//        sign *= CheckSign(tmp[sp], p);
//        tmp[sp][p] = 1;
//        Jnew = STLBitsetDeterminant(tmp[0], tmp[1]);
//        timer_off("2PO");
//        return sign;
//    } else {
//        timer_off("2PO");
//        return 0.0;
//    }
//}

// void FCI_MO::FormCumulant3(CI_RDMS& ci_rdms, d6& AAA, d6& AAB, d6& ABB, d6& BBB, string& DC) {
//    timer_on("FORM 3-Cumulant");
//    Timer tL3;
//    std::string str = "Forming Lambda3";
//    outfile->Printf("\n  %-35s ...", str.c_str());

//    size_t dim = na_ * na_ * na_ * na_ * na_ * na_;
//    std::vector<double> tpdm_aaa(dim, 0.0);
//    std::vector<double> tpdm_aab(dim, 0.0);
//    std::vector<double> tpdm_abb(dim, 0.0);
//    std::vector<double> tpdm_bbb(dim, 0.0);

//    ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

//    FormCumulant3AAA(tpdm_aaa, tpdm_bbb, AAA, BBB, DC);
//    FormCumulant3AAB(tpdm_aab, tpdm_abb, AAB, ABB, DC);
//    //    fill_cumulant3();

//    outfile->Printf("  Done. Timing %15.6f s", tL3.get());
//    timer_off("FORM 3-Cumulant");
//}

// void FCI_MO::FormCumulant3AAA(const std::vector<double>& tpdm_aaa,
//                              const std::vector<double>& tpdm_bbb, d6& AAA, d6& BBB, string& DC) {
//    size_t dim2 = na_ * na_;
//    size_t dim3 = na_ * dim2;
//    size_t dim4 = na_ * dim3;
//    size_t dim5 = na_ * dim4;

//    for (size_t p = 0; p != na_; ++p) {
//        size_t np = idx_a_[p];
//        for (size_t q = p + 1; q != na_; ++q) {
//            size_t nq = idx_a_[q];
//            for (size_t r = q + 1; r != na_; ++r) {
//                size_t nr = idx_a_[r];
//                for (size_t s = 0; s != na_; ++s) {
//                    size_t ns = idx_a_[s];
//                    for (size_t t = s + 1; t != na_; ++t) {
//                        size_t nt = idx_a_[t];
//                        for (size_t u = t + 1; u != na_; ++u) {
//                            size_t nu = idx_a_[u];

//                            if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]
//                            ^
//                                 sym_active_[t] ^ sym_active_[u]) != 0)
//                                continue;

//                            if (DC == "MK") {
//                                size_t index =
//                                    p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * na_ + u;

//                                AAA[p][q][r][s][t][u] += tpdm_aaa[index];
//                                BBB[p][q][r][s][t][u] += tpdm_bbb[index];
//                            }

//                            AAA[p][q][r][s][t][u] -= P3DDD(Da_, np, nq, nr, ns, nt, nu);
//                            AAA[p][q][r][s][t][u] -= P3DC(Da_, L2aa_, p, q, r, s, t, u);

//                            BBB[p][q][r][s][t][u] -= P3DDD(Db_, np, nq, nr, ns, nt, nu);
//                            BBB[p][q][r][s][t][u] -= P3DC(Db_, L2bb_, p, q, r, s, t, u);

//                            size_t cop[] = {p, q, r};
//                            size_t aop[] = {s, t, u};
//                            int P1 = 1;
//                            do {
//                                int P2 = 1;
//                                do {
//                                    double sign = pow(-1.0, int(P1 / 2) + int(P2 / 2));
//                                    AAA[cop[0]][cop[1]][cop[2]][aop[0]][aop[1]][aop[2]] =
//                                        sign * AAA[p][q][r][s][t][u];
//                                    BBB[cop[0]][cop[1]][cop[2]][aop[0]][aop[1]][aop[2]] =
//                                        sign * BBB[p][q][r][s][t][u];
//                                    ++P2;
//                                } while (std::next_permutation(aop, aop + 3));
//                                ++P1;
//                            } while (std::next_permutation(cop, cop + 3));
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

// void FCI_MO::FormCumulant3AAB(const std::vector<double>& tpdm_aab,
//                              const std::vector<double>& tpdm_abb, d6& AAB, d6& ABB, string& DC) {
//    size_t dim2 = na_ * na_;
//    size_t dim3 = na_ * dim2;
//    size_t dim4 = na_ * dim3;
//    size_t dim5 = na_ * dim4;

//    for (size_t p = 0; p != na_; ++p) {
//        size_t np = idx_a_[p];
//        for (size_t q = p + 1; q != na_; ++q) {
//            size_t nq = idx_a_[q];
//            for (size_t r = 0; r != na_; ++r) {
//                size_t nr = idx_a_[r];
//                for (size_t s = 0; s != na_; ++s) {
//                    size_t ns = idx_a_[s];
//                    for (size_t t = s + 1; t != na_; ++t) {
//                        size_t nt = idx_a_[t];
//                        for (size_t u = 0; u != na_; ++u) {
//                            size_t nu = idx_a_[u];

//                            if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]
//                            ^
//                                 sym_active_[t] ^ sym_active_[u]) != 0)
//                                continue;

//                            if (DC == "MK") {
//                                size_t index =
//                                    p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * na_ + u;
//                                AAB[p][q][r][s][t][u] += tpdm_aab[index];

//                                index = r * dim5 + p * dim4 + q * dim3 + u * dim2 + s * na_ + t;
//                                ABB[r][p][q][u][s][t] += tpdm_abb[index];
//                            }

//                            AAB[p][q][r][s][t][u] -= (Da_[np][ns] * Da_[nq][nt] * Db_[nr][nu] -
//                                                      Da_[nq][ns] * Da_[np][nt] * Db_[nr][nu]);
//                            AAB[p][q][r][s][t][u] -=
//                                (Da_[np][ns] * L2ab_[q][r][t][u] - Da_[np][nt] *
//                                L2ab_[q][r][s][u]);
//                            AAB[p][q][r][s][t][u] -=
//                                (Da_[nq][nt] * L2ab_[p][r][s][u] - Da_[nq][ns] *
//                                L2ab_[p][r][t][u]);
//                            AAB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2aa_[p][q][s][t]);
//                            AAB[q][p][r][s][t][u] -= AAB[p][q][r][s][t][u];
//                            AAB[p][q][r][t][s][u] -= AAB[p][q][r][s][t][u];
//                            AAB[q][p][r][t][s][u] += AAB[p][q][r][s][t][u];

//                            ABB[r][p][q][u][s][t] -= (Db_[np][ns] * Db_[nq][nt] * Da_[nr][nu] -
//                                                      Db_[nq][ns] * Db_[np][nt] * Da_[nr][nu]);
//                            ABB[r][p][q][u][s][t] -=
//                                (Db_[np][ns] * L2ab_[r][q][u][t] - Db_[np][nt] *
//                                L2ab_[r][q][u][s]);
//                            ABB[r][p][q][u][s][t] -=
//                                (Db_[nq][nt] * L2ab_[r][p][u][s] - Db_[nq][ns] *
//                                L2ab_[r][p][u][t]);
//                            ABB[r][p][q][u][s][t] -= (Da_[nr][nu] * L2bb_[p][q][s][t]);
//                            ABB[r][q][p][u][s][t] -= ABB[r][p][q][u][s][t];
//                            ABB[r][p][q][u][t][s] -= ABB[r][p][q][u][s][t];
//                            ABB[r][q][p][u][t][s] += ABB[r][p][q][u][s][t];
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

// void FCI_MO::FormCumulant3_DIAG(const vecdet& dets, const int& root, d6& AAA, d6& AAB, d6& ABB,
//                                d6& BBB) {
//    timer_on("FORM 3-Cumulant");
//    Timer tL3;
//    std::string str = "Forming Lambda3";
//    outfile->Printf("\n  %-35s ...", str.c_str());
//    for (size_t p = 0; p < na_; ++p) {
//        size_t np = idx_a_[p];
//        for (size_t q = 0; q < na_; ++q) {
//            size_t nq = idx_a_[q];
//            for (size_t r = 0; r < na_; ++r) {
//                size_t nr = idx_a_[r];

//                size_t size = dets.size();
//                for (size_t ket = 0; ket != size; ++ket) {
//                    STLBitsetDeterminant Jaaa(vector<bool>(2 * ncmo_)),
//                        Jaab(vector<bool>(2 * ncmo_)), Jabb(vector<bool>(2 * ncmo_)),
//                        Jbbb(vector<bool>(2 * ncmo_));
//                    double aaa = 1.0, aab = 1.0, abb = 1.0, bbb = 1.0,
//                           vket = (eigen_[root].first)->get(ket);
//                    ;
//                    aaa *= ThreeOP(dets[ket], Jaaa, p, 0, q, 0, r, 0, p, 0, q, 0, r, 0) * vket;
//                    aab *= ThreeOP(dets[ket], Jaab, p, 0, q, 0, r, 1, p, 0, q, 0, r, 1) * vket;
//                    abb *= ThreeOP(dets[ket], Jabb, p, 0, q, 1, r, 1, p, 0, q, 1, r, 1) * vket;
//                    bbb *= ThreeOP(dets[ket], Jbbb, p, 1, q, 1, r, 1, p, 1, q, 1, r, 1) * vket;

//                    for (size_t bra = 0; bra != size; ++bra) {
//                        double vbra = (eigen_[root].first)->get(bra);
//                        AAA[p][q][r][p][q][r] += aaa * (dets[bra] == Jaaa) * vbra;
//                        AAB[p][q][r][p][q][r] += aab * (dets[bra] == Jaab) * vbra;
//                        ABB[p][q][r][p][q][r] += abb * (dets[bra] == Jabb) * vbra;
//                        BBB[p][q][r][p][q][r] += bbb * (dets[bra] == Jbbb) * vbra;
//                    }
//                }

//                AAA[p][q][r][p][q][r] -= P3DDD(Da_, np, nq, nr, np, nq, nr);
//                AAA[p][q][r][p][q][r] -= P3DC(Da_, L2aa_, p, q, r, p, q, r);
//                AAA[p][q][r][p][r][q] -= AAA[p][q][r][p][q][r];
//                AAA[p][q][r][q][p][r] -= AAA[p][q][r][p][q][r];
//                AAA[p][q][r][q][r][p] = AAA[p][q][r][p][q][r];
//                AAA[p][q][r][r][p][q] = AAA[p][q][r][p][q][r];
//                AAA[p][q][r][r][q][p] -= AAA[p][q][r][p][q][r];

//                AAB[p][q][r][p][q][r] -= (Da_[np][np] * Da_[nq][nq] * Db_[nr][nr] -
//                                          Da_[nq][np] * Da_[np][nq] * Db_[nr][nr]);
//                AAB[p][q][r][p][q][r] -=
//                    (Da_[np][np] * L2ab_[q][r][q][r] - Da_[np][nq] * L2ab_[q][r][p][r]);
//                AAB[p][q][r][p][q][r] -=
//                    (Da_[nq][nq] * L2ab_[p][r][p][r] - Da_[nq][np] * L2ab_[p][r][q][r]);
//                AAB[p][q][r][p][q][r] -= (Db_[nr][nr] * L2aa_[p][q][p][q]);
//                AAB[p][q][r][q][p][r] -= AAB[p][q][r][p][q][r];

//                ABB[p][q][r][p][q][r] -= (Da_[np][np] * Db_[nq][nq] * Db_[nr][nr] -
//                                          Da_[np][np] * Db_[nr][nq] * Db_[nq][nr]);
//                ABB[p][q][r][p][q][r] -=
//                    (Db_[nq][nq] * L2ab_[p][r][p][r] - Db_[nq][nr] * L2ab_[p][r][p][q]);
//                ABB[p][q][r][p][q][r] -=
//                    (Db_[nr][nr] * L2ab_[p][q][p][q] - Db_[nr][nq] * L2ab_[p][q][p][r]);
//                ABB[p][q][r][p][q][r] -= (Da_[np][np] * L2bb_[q][r][q][r]);
//                ABB[p][r][q][p][q][r] -= ABB[p][q][r][p][q][r];

//                BBB[p][q][r][p][q][r] -= P3DDD(Db_, np, nq, nr, np, nq, nr);
//                BBB[p][q][r][p][q][r] -= P3DC(Db_, L2bb_, p, q, r, p, q, r);
//                BBB[p][q][r][p][r][q] -= BBB[p][q][r][p][q][r];
//                BBB[p][q][r][q][p][r] -= BBB[p][q][r][p][q][r];
//                BBB[p][q][r][q][r][p] = BBB[p][q][r][p][q][r];
//                BBB[p][q][r][r][p][q] = BBB[p][q][r][p][q][r];
//                BBB[p][q][r][r][q][p] -= BBB[p][q][r][p][q][r];
//            }
//        }
//    }
//    fill_cumulant3();
//    outfile->Printf("  Done. Timing %15.6f s", tL3.get());
//    timer_off("FORM 3-Cumulant");
//}

void FCI_MO::print3PDC(const string& str, const d6& ThreePDC, const int& PRINT) {
    timer_on("PRINT 3-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    size_t size = ThreePDC.size();
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != size; ++j) {
            for (size_t k = 0; k != size; ++k) {
                for (size_t l = 0; l != size; ++l) {
                    for (size_t m = 0; m != size; ++m) {
                        for (size_t n = 0; n != size; ++n) {
                            if (std::fabs(ThreePDC[i][j][k][l][m][n]) > 1.0e-15) {
                                ++count;
                                if (PRINT > 3)
                                    outfile->Printf("\n  Lambda "
                                                    "[%3lu][%3lu][%3lu][%3lu][%"
                                                    "3lu][%3lu] = %18.15lf",
                                                    i, j, k, l, m, n, ThreePDC[i][j][k][l][m][n]);
                            }
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 3-Cumulant");
}

// double FCI_MO::ThreeOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t&
// p,
//                       const bool& sp, const size_t& q, const bool& sq, const size_t& r,
//                       const bool& sr, const size_t& s, const bool& ss, const size_t& t,
//                       const bool& st, const size_t& u, const bool& su) {
//    timer_on("3PO");
//    std::vector<vector<bool>> tmp;
//    tmp.push_back(J.get_alfa_bits_vector_bool());
//    tmp.push_back(J.get_beta_bits_vector_bool());

//    double sign = 1.0;

//    if (tmp[ss][s]) {
//        sign *= CheckSign(tmp[ss], s);
//        tmp[ss][s] = 0;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }

//    if (tmp[st][t]) {
//        sign *= CheckSign(tmp[st], t);
//        tmp[st][t] = 0;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }

//    if (tmp[su][u]) {
//        sign *= CheckSign(tmp[su], u);
//        tmp[su][u] = 0;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }

//    if (!tmp[sr][r]) {
//        sign *= CheckSign(tmp[sr], r);
//        tmp[sr][r] = 1;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }

//    if (!tmp[sq][q]) {
//        sign *= CheckSign(tmp[sq], q);
//        tmp[sq][q] = 1;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }

//    if (!tmp[sp][p]) {
//        sign *= CheckSign(tmp[sp], p);
//        tmp[sp][p] = 1;
//        Jnew = STLBitsetDeterminant(tmp[0], tmp[1]);
//        timer_off("3PO");
//        return sign;
//    } else {
//        timer_off("3PO");
//        return 0.0;
//    }
//}

void FCI_MO::print_Fock(const string& spin, const d2& Fock) {
    string name = "Fock " + spin;
    outfile->Printf("  ==> %s <==\n\n", name.c_str());

    // print Fock block
    auto print_Fock_block = [&](const string& name1, const string& name2,
                                const std::vector<size_t>& idx1, const std::vector<size_t>& idx2) {
        size_t dim1 = idx1.size();
        size_t dim2 = idx2.size();
        string bname = name1 + "-" + name2;

        Matrix F(bname, dim1, dim2);
        for (size_t i = 0; i < dim1; ++i) {
            size_t ni = idx1[i];
            for (size_t j = 0; j < dim2; ++j) {
                size_t nj = idx2[j];
                F.set(i, j, Fock[ni][nj]);
            }
        }

        F.print();

        if (dim1 != dim2) {
            string bnamer = name2 + "-" + name1;
            Matrix Fr(bnamer, dim2, dim1);
            for (size_t i = 0; i < dim2; ++i) {
                size_t ni = idx2[i];
                for (size_t j = 0; j < dim1; ++j) {
                    size_t nj = idx1[j];
                    Fr.set(i, j, Fock[ni][nj]);
                }
            }

            SharedMatrix FT = Fr.transpose();
            for (size_t i = 0; i < dim1; ++i) {
                for (size_t j = 0; j < dim2; ++j) {
                    double diff = FT->get(i, j) - F.get(i, j);
                    FT->set(i, j, diff);
                }
            }
            if (FT->rms() > fcheck_threshold_) {
                outfile->Printf("  Warning: %s not symmetric for %s and %s blocks\n", name.c_str(),
                                bname.c_str(), bnamer.c_str());
                Fr.print();
            }
        }
    };

    // diagonal blocks
    print_Fock_block("C", "C", idx_c_, idx_c_);
    print_Fock_block("V", "V", idx_v_, idx_v_);

    std::vector<size_t> idx_ah, idx_ap;
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        for (int i = 0; i < ah_.size(); ++i) {
            idx_ah.push_back(idx_a_[ah_[i]]);
        }
        for (int i = 0; i < ap_.size(); ++i) {
            idx_ap.push_back(idx_a_[ap_[i]]);
        }
        print_Fock_block("AH", "AH", idx_ah, idx_ah);
        print_Fock_block("AP", "AP", idx_ap, idx_ap);
    } else {
        print_Fock_block("A", "A", idx_a_, idx_a_);
    }

    // off-diagonal blocks
    print_Fock_block("C", "A", idx_c_, idx_a_);
    print_Fock_block("C", "V", idx_c_, idx_v_);
    print_Fock_block("A", "V", idx_a_, idx_v_);
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        print_Fock_block("AH", "AP", idx_ah, idx_ap);
    }
}

void FCI_MO::Form_Fock(d2& A, d2& B) {
    timer_on("Form Fock");
    compute_Fock_ints();

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            A[p][q] = integral_->get_fock_a(p, q);
            B[p][q] = integral_->get_fock_b(p, q);
        }
    }

    //    ambit::Tensor Fa =
    //    ambit::Tensor::build(ambit::CoreTensor,"Fa",{ncmo_,ncmo_});
    //    ambit::Tensor Fb =
    //    ambit::Tensor::build(ambit::CoreTensor,"Fb",{ncmo_,ncmo_});

    //    Fa.iterate([&](const std::vector<size_t>& i,double& value){
    //        value = integral_->oei_a(i[0],i[1]);
    //        for(const size_t& c: idx_c_){
    //            value += integral_->aptei_aa(i[0],c,i[1],c);
    //            value += integral_->aptei_ab(i[0],c,i[1],c);
    //        }
    //    });
    //    Fb.iterate([&](const std::vector<size_t>& i,double& value){
    //        value = integral_->oei_b(i[0],i[1]);
    //        for(const size_t& c: idx_c_){
    //            value += integral_->aptei_bb(i[0],c,i[1],c);
    //            value += integral_->aptei_ab(c,i[0],c,i[1]);
    //        }
    //    });

    //    std::vector<size_t> idx_corr (ncmo_);
    //    std::iota(idx_corr.begin(), idx_corr.end(), 0);
    //    ambit::Tensor V =
    //    integral_->aptei_aa_block(idx_corr,idx_a_,idx_corr,idx_a_);
    //    Fa("pq") += V("puqv") * L1a("vu");

    //    V = integral_->aptei_ab_block(idx_corr,idx_a_,idx_corr,idx_a_);
    //    Fa("pq") += V("puqv") * L1b("vu");

    //    V = integral_->aptei_ab_block(idx_a_,idx_corr,idx_a_,idx_corr);
    //    Fb("pq") += V("upvq") * L1a("vu");

    //    V = integral_->aptei_bb_block(idx_corr,idx_a_,idx_corr,idx_a_);
    //    Fb("pq") += V("puqv") * L1b("vu");

    //    for(size_t p = 0; p < ncmo_; ++p){
    //        for(size_t q = 0; q < ncmo_; ++q){
    //            A[p][q] = Fa.data()[p * ncmo_ + q];
    //            B[p][q] = Fb.data()[p * ncmo_ + q];
    //        }
    //    }
    timer_off("Form Fock");
}

void FCI_MO::compute_Fock_ints() {
    Timer tfock;
    std::string str = "Forming generalized Fock matrix";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
    }

    SharedMatrix DaM(new Matrix("DaM", ncmo_, ncmo_));
    SharedMatrix DbM(new Matrix("DbM", ncmo_, ncmo_));
    for (size_t m = 0; m < nc_; m++) {
        size_t nm = idx_c_[m];
        for (size_t n = 0; n < nc_; n++) {
            size_t nn = idx_c_[n];
            DaM->set(nm, nn, Da_[nm][nn]);
            DbM->set(nm, nn, Db_[nm][nn]);
        }
    }
    for (size_t u = 0; u < na_; u++) {
        size_t nu = idx_a_[u];
        for (size_t v = 0; v < na_; v++) {
            size_t nv = idx_a_[v];
            DaM->set(nu, nv, Da_[nu][nv]);
            DbM->set(nu, nv, Db_[nu][nv]);
        }
    }
    integral_->make_fock_matrix(DaM, DbM);

    if (!quiet_) {
        outfile->Printf("  Done. Timing %15.6f s", tfock.get());
    }
}

void FCI_MO::Check_Fock(const d2& A, const d2& B, const double& E, size_t& count) {
    timer_on("Check Fock");
    Timer tfock;
    std::string str = "Checking Fock matrices (Fa, Fb)";
    if (!quiet_) {
        outfile->Printf("\n  %-35s ...", str.c_str());
        outfile->Printf("\n  Nonzero criteria: > %.2E", E);
    }
    Check_FockBlock(A, B, E, count, nc_, idx_c_, "CORE");
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        std::vector<size_t> idx_ah, idx_ap;
        for (int i = 0; i < ah_.size(); ++i) {
            idx_ah.push_back(idx_a_[ah_[i]]);
        }
        for (int i = 0; i < ap_.size(); ++i) {
            idx_ap.push_back(idx_a_[ap_[i]]);
        }
        Check_FockBlock(A, B, E, count, ah_.size(), idx_ah, "ACT_H");
        Check_FockBlock(A, B, E, count, ap_.size(), idx_ap, "ACT_P");
    } else {
        Check_FockBlock(A, B, E, count, na_, idx_a_, "ACTIVE");
    }
    Check_FockBlock(A, B, E, count, nv_, idx_v_, "VIRTUAL");
    str = "Done checking Fock matrices.";
    if (!quiet_) {
        outfile->Printf("\n  %-47s", str.c_str());
        outfile->Printf("Timing %15.6f s", tfock.get());
        outfile->Printf("\n");
    }
    timer_off("Check Fock");
}

void FCI_MO::Check_FockBlock(const d2& A, const d2& B, const double& E, size_t& count,
                             const size_t& dim, const std::vector<size_t>& idx, const string& str) {
    double maxa = 0.0, maxb = 0.0;
    size_t a = 0, b = 0;
    for (size_t p = 0; p < dim; ++p) {
        size_t np = idx[p];
        for (size_t q = 0; q < dim; ++q) {
            size_t nq = idx[q];
            if (np != nq) {
                if (std::fabs(A[np][nq]) > E) {
                    ++a;
                    maxa = (std::fabs(A[np][nq]) > maxa) ? std::fabs(A[np][nq]) : maxa;
                }
                if (std::fabs(B[np][nq]) > E) {
                    ++b;
                    maxb = (std::fabs(B[np][nq]) > maxb) ? std::fabs(B[np][nq]) : maxb;
                }
            }
        }
    }
    count += a + b;
    if (!quiet_) {
        if (a == 0) {
            outfile->Printf("\n  Fa_%-7s block is diagonal.", str.c_str());
        } else {
            outfile->Printf("\n  Warning: Fa_%-7s NOT diagonal!", str.c_str());
            outfile->Printf("\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", a, maxa);
        }
        if (b == 0) {
            outfile->Printf("\n  Fb_%-7s block is diagonal.", str.c_str());
        } else {
            outfile->Printf("\n  Warning: Fb_%-7s NOT diagonal!", str.c_str());
            outfile->Printf("\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", b, maxb);
        }
    }
}

void FCI_MO::BD_Fock(const d2& Fa, const d2& Fb, SharedMatrix& Ua, SharedMatrix& Ub,
                     const string& name) {
    timer_on("Block Diagonal 2D Matrix");
    Timer tbdfock;
    std::string str = "Diagonalizing " + name;
    outfile->Printf("\n  %-35s ...", str.c_str());

    // separate Fock to core, active, virtual blocks
    SharedMatrix Fc_a(new Matrix("Fock core alpha", core_, core_));
    SharedMatrix Fc_b(new Matrix("Fock core beta", core_, core_));
    SharedMatrix Fv_a(new Matrix("Fock virtual alpha", virtual_, virtual_));
    SharedMatrix Fv_b(new Matrix("Fock virtual beta", virtual_, virtual_));
    // core and virtual
    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        h = static_cast<int>(h);
        for (size_t i = 0; i < core_[h]; ++i) {
            for (size_t j = 0; j < core_[h]; ++j) {
                Fc_a->set(h, i, j, Fa[offset + i][offset + j]);
                Fc_b->set(h, i, j, Fb[offset + i][offset + j]);
            }
        }
        offset += core_[h] + active_[h];

        for (size_t a = 0; a < virtual_[h]; ++a) {
            for (size_t b = 0; b < virtual_[h]; ++b) {
                Fv_a->set(h, a, b, Fa[offset + a][offset + b]);
                Fv_b->set(h, a, b, Fb[offset + a][offset + b]);
            }
        }
        offset += virtual_[h];
    }
    // active
    SharedMatrix Fa_a, Fa_b, Fao_a, Fao_b, Fav_a, Fav_b;
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        Fao_a = SharedMatrix(new Matrix("Fock active hole alpha", active_h_, active_h_));
        Fao_b = SharedMatrix(new Matrix("Fock active hole beta", active_h_, active_h_));
        Fav_a = SharedMatrix(new Matrix("Fock active particle alpha", active_p_, active_p_));
        Fav_b = SharedMatrix(new Matrix("Fock active particle beta", active_p_, active_p_));
        for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
            h = static_cast<int>(h);
            offset += core_[h];
            // active occupied
            for (int u = 0; u < active_h_[h]; ++u) {
                for (int v = 0; v < active_h_[h]; ++v) {
                    Fao_a->set(h, u, v, Fa[offset + u][offset + v]);
                    Fao_b->set(h, u, v, Fb[offset + u][offset + v]);
                }
            }
            // active virtual
            for (int u = active_h_[h]; u < active_[h]; ++u) {
                int nu = u - active_h_[h];
                for (int v = active_h_[h]; v < active_[h]; ++v) {
                    int nv = v - active_h_[h];
                    Fav_a->set(h, nu, nv, Fa[offset + u][offset + v]);
                    Fav_b->set(h, nu, nv, Fb[offset + u][offset + v]);
                }
            }
            offset += active_[h] + virtual_[h];
        }
    } else {
        Fa_a = SharedMatrix(new Matrix("Fock active alpha", active_, active_));
        Fa_b = SharedMatrix(new Matrix("Fock active beta", active_, active_));
        for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
            h = static_cast<int>(h);
            offset += core_[h];
            for (int u = 0; u < active_[h]; ++u) {
                for (int v = 0; v < active_[h]; ++v) {
                    Fa_a->set(h, u, v, Fa[offset + u][offset + v]);
                    Fa_b->set(h, u, v, Fb[offset + u][offset + v]);
                }
            }
            offset += active_[h] + virtual_[h];
        }
    }

    // diagonalize Fock blocks
    std::vector<SharedMatrix> blocks;
    std::vector<SharedMatrix> evecs;
    std::vector<SharedVector> evals;
    if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
        blocks = {Fc_a, Fc_b, Fv_a, Fv_b, Fao_a, Fao_b, Fav_a, Fav_b};
    } else {
        blocks = {Fc_a, Fc_b, Fv_a, Fv_b, Fa_a, Fa_b};
    }
    for (auto F : blocks) {
        std::string name = "U for " + F->name();
        SharedMatrix U(new Matrix(name, F->rowspi(), F->colspi()));
        SharedVector lambda(new Vector("lambda", F->rowspi()));
        F->diagonalize(U, lambda);
        evecs.push_back(U);
        evals.push_back(lambda);

        //        U->eivprint(lambda);
        //        SharedMatrix X = Matrix::triplet(U,F,U,true,false,false);
        //        X->print();
    }

    // fill in the unitary rotation
    for (int h = 0; h < nirrep_; ++h) {
        size_t offset = 0;

        // frozen core
        for (size_t i = 0; i < frzcpi_[h]; ++i) {
            Ua->set(h, i, i, 1.0);
            Ub->set(h, i, i, 1.0);
        }
        offset += frzcpi_[h];

        // core
        for (size_t i = 0; i < core_[h]; ++i) {
            for (size_t j = 0; j < core_[h]; ++j) {
                Ua->set(h, offset + i, offset + j, evecs[0]->get(h, i, j));
                Ub->set(h, offset + i, offset + j, evecs[1]->get(h, i, j));
            }
        }
        offset += core_[h];

        // active
        if (active_space_type_ == "CIS" || active_space_type_ == "CISD") {
            for (int u = 0; u < active_h_[h]; ++u) {
                for (int v = 0; v < active_h_[h]; ++v) {
                    Ua->set(h, offset + u, offset + v, evecs[4]->get(h, u, v));
                    Ub->set(h, offset + u, offset + v, evecs[5]->get(h, u, v));
                }
            }
            for (int u = active_h_[h]; u < active_[h]; ++u) {
                int nu = u - active_h_[h];
                for (int v = active_h_[h]; v < active_[h]; ++v) {
                    int nv = v - active_h_[h];
                    Ua->set(h, offset + u, offset + v, evecs[6]->get(h, nu, nv));
                    Ub->set(h, offset + u, offset + v, evecs[7]->get(h, nu, nv));
                }
            }
        } else {
            for (int u = 0; u < active_[h]; ++u) {
                for (int v = 0; v < active_[h]; ++v) {
                    Ua->set(h, offset + u, offset + v, evecs[4]->get(h, u, v));
                    Ub->set(h, offset + u, offset + v, evecs[5]->get(h, u, v));
                }
            }
        }
        offset += active_[h];

        // virtual
        for (size_t a = 0; a < virtual_[h]; ++a) {
            for (size_t b = 0; b < virtual_[h]; ++b) {
                Ua->set(h, offset + a, offset + b, evecs[2]->get(h, a, b));
                Ub->set(h, offset + a, offset + b, evecs[3]->get(h, a, b));
            }
        }
        offset += virtual_[h];

        // frozen virtual
        for (size_t i = 0; i < frzvpi_[h]; ++i) {
            size_t j = i + offset;
            Ua->set(h, j, j, 1.0);
            Ub->set(h, j, j, 1.0);
        }
        offset += frzvpi_[h];
    }

    outfile->Printf("  Done. Timing %15.6f s\n", tbdfock.get());
    timer_off("Block Diagonal 2D Matrix");
}

void FCI_MO::compute_permanent_dipole() {

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::string irrep_symbol = ct.gamma(root_sym_).symbol();
    std::string title = "Permanent Dipole Moments (" + irrep_symbol + ")";
    print_h2(title);
    outfile->Printf("\n  Only print nonzero (> 1.0e-5) elements.");

    // obtain AO dipole from ForteIntegrals
    std::vector<SharedMatrix> aodipole_ints = integral_->AOdipole_ints();

    // Nuclear dipole contribution
    SharedVector ndip =
        DipoleInt::nuclear_contribution(Process::environment.molecule(), Vector3(0.0, 0.0, 0.0));

    // SO to AO transformer
    SharedMatrix sotoao(this->aotoso()->transpose());

    // fill the density according to point group to a SharedMatrix
    auto fill_density = [&](const std::vector<double>& vec, const SharedMatrix& mat) {
        mat->zero();
        size_t offset = 0;

        for (int h = 0; h < nirrep_; ++h) {
            // frozen core
            for (size_t i = 0; i < frzcpi_[h]; ++i) {
                mat->set(h, i, i, 1.0);
            }

            // restricted core
            size_t offset1 = frzcpi_[h];
            for (size_t i = 0; i < core_[h]; ++i) {
                size_t ni = i + offset1;
                mat->set(h, ni, ni, 1.0);
            }

            // active
            offset1 += core_[h];
            for (size_t u = 0; u < active_[h]; ++u) {
                size_t mu = u + offset;
                size_t nu = u + offset1;

                for (size_t v = 0; v < active_[h]; ++v) {
                    size_t mv = v + offset;
                    size_t nv = v + offset1;

                    mat->set(h, nu, nv, vec[mu * na_ + mv]);
                }
            }
            offset += active_[h];
        }
    };

    // prepare eigen vectors for ci_rdm
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
    for (int i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }

    // loop over states
    for (int A = 0; A < nroot_; ++A) {
        std::string trans_name = std::to_string(A) + " -> " + std::to_string(A);

        CI_RDMS ci_rdms(options_, fci_ints_, determinant_, evecs, A, A);
        std::vector<double> opdm_a(na_ * na_, 0.0);
        std::vector<double> opdm_b(na_ * na_, 0.0);
        ci_rdms.compute_1rdm(opdm_a, opdm_b);

        SharedMatrix SOdens(new Matrix("SO density " + trans_name, nmopi_, nmopi_));
        fill_density(opdm_a, SOdens);
        SOdens->back_transform(this->Ca());

        size_t nao = sotoao->coldim(0);
        SharedMatrix AOdens(new Matrix("AO density " + trans_name, nao, nao));
        AOdens->remove_symmetry(SOdens, sotoao);

        std::vector<double> de(4, 0.0);
        for (int i = 0; i < 3; ++i) {
            de[i] = 2.0 * AOdens->vector_dot(aodipole_ints[i]); // 2.0 for beta spin
            de[i] += ndip->get(i);                              // add nuclear contributions
            de[3] += de[i] * de[i];
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

void FCI_MO::compute_transition_dipole() {

    //    if(nirrep_ != 1){
    //        outfile->Printf("\n  Computing transition dipole moments in %s
    //        symmetry.",
    //                        Process::environment.molecule()->sym_label().c_str());
    //        outfile->Printf("\n  Currently only support transitions with the
    //        same irrep.");
    //        outfile->Printf("\n  Please set molecular symmetry to C1.\n");
    //        return;
    //    }

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::string irrep_symbol = ct.gamma(root_sym_).symbol();
    std::stringstream title;
    title << "Transition Dipole Moments (" << irrep_symbol << " -> " << irrep_symbol << ")";
    print_h2(title.str());
    outfile->Printf("\n  Only print nonzero (> 1.0e-5) elements.");

    // obtain AO dipole from libmints
    std::vector<SharedMatrix> aodipole_ints = integral_->AOdipole_ints();

    // SO to AO transformer
    SharedMatrix sotoao(this->aotoso()->transpose());

    //    // obtain SO dipole from libmints
    //    std::vector<SharedMatrix> dipole_ints;
    //    for(const std::string& direction: {"X","Y","Z"}){
    //        std::string name = "SO Dipole" + direction;
    //        dipole_ints.push_back(SharedMatrix(new Matrix(name, this->nsopi(),
    //        this->nsopi()) ));
    //    }

    //    std::shared_ptr<BasisSet> basisset = this->basisset();
    //    std::shared_ptr<IntegralFactory> ints =
    //    std::shared_ptr<IntegralFactory>(
    //                new IntegralFactory(basisset,basisset,basisset,basisset));
    //    std::shared_ptr<OneBodySOInt> sodOBI(ints->so_dipole());

    //    Vector3 origin (0.0, 0.0, 0.0);
    //    sodOBI->ob()->set_origin(origin);
    //    sodOBI->compute(dipole_ints);

    //    // transform SO dipole to MO dipole
    //    for(SharedMatrix& dipole: dipole_ints){
    //        dipole->transform(this->Ca());
    //    }

    // fill the density according to point group to a SharedMatrix
    auto fill_density = [&](const std::vector<double>& vec, const SharedMatrix& mat) {

        size_t offset = 0;
        mat->zero();

        for (int h = 0; h < nirrep_; ++h) {

            // active only
            size_t offset1 = frzcpi_[h] + core_[h];
            for (size_t u = 0; u < active_[h]; ++u) {
                size_t mu = u + offset;
                size_t nu = u + offset1;

                for (size_t v = 0; v < active_[h]; ++v) {
                    size_t mv = v + offset;
                    size_t nv = v + offset1;

                    mat->set(h, nu, nv, vec[mu * na_ + mv]);
                }
            }
            offset += active_[h];
        }
    };

    // prepare eigen vectors for ci_rdm
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
    for (int i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }

    // loop over states of the same symmetry
    trans_dipole_.clear();
    for (int A = 0; A < nroot_; ++A) {
        for (int B = A + 1; B < nroot_; ++B) {
            std::string trans_name = std::to_string(A) + " -> " + std::to_string(B);

            CI_RDMS ci_rdms(options_, fci_ints_, determinant_, evecs, A, B);
            std::vector<double> opdm_a(na_ * na_, 0.0);
            std::vector<double> opdm_b(na_ * na_, 0.0);
            ci_rdms.compute_1rdm(opdm_a, opdm_b);

            SharedMatrix SOtransD(
                new Matrix("SO transition density " + trans_name, nmopi_, nmopi_));
            fill_density(opdm_a, SOtransD);
            SOtransD->back_transform(this->Ca());

            size_t nao = sotoao->coldim(0);
            SharedMatrix AOtransD(new Matrix("AO transition density " + trans_name, nao, nao));
            AOtransD->remove_symmetry(SOtransD, sotoao);

            std::vector<double> de(4, 0.0);
            for (int i = 0; i < 3; ++i) {
                de[i] = 2.0 * AOtransD->vector_dot(aodipole_ints[i]); // 2.0 for beta spin
                //                de[i] = 2.0 *
                //                        MOtransD->vector_dot(dipole_ints[i]);
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
    //        CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,0,A);
    //        std::vector<double> opdm_a (na_ * na_, 0.0);
    //        std::vector<double> opdm_b (na_ * na_, 0.0);
    //        ci_rdms.compute_1rdm(opdm_a, opdm_b);

    //        SharedMatrix transD (new Matrix("MO transition density 0 -> " +
    //        std::to_string(A), nmopi_, nmopi_));
    //        symmetrize_density(opdm_a, transD);
    //        transD->back_transform(this->Ca());

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

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
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
FCI_MO::compute_relaxed_dm(const std::vector<double>& dm0, std::vector<BlockedTensor>& dm1,
                           std::vector<BlockedTensor>& dm2) {
    std::map<std::string, std::vector<double>> out;

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> mt{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    double dm0_sum = std::fabs(dm0[0]) + std::fabs(dm0[1]) + std::fabs(dm0[2]);
    std::vector<bool> do_dm;
    for (int z = 0; z < 3; ++z) {
        do_dm.push_back(std::fabs(dm0[z]) > 1.0e-12 ? true : false);
    }

    std::string pg = ct.symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi, const int& root, const int& irrep) {
        std::string symbol = ct.gamma(irrep).symbol();
        std::stringstream name_ss;
        name_ss << std::setw(2) << root << " " << std::setw(7) << mt[multi - 1] << " "
                << std::setw(width) << symbol;
        return name_ss.str();
    };

    // if SS, read from determinant_ and eigen_; otherwise, read from p_spaces_ and eigens_
    if (sa_info_.size() == 0) {
        std::string name = generate_name(multi_, root_, root_sym_);
        std::vector<double> dm(3, 0.0);

        // prepare CI_RDMS
        int dim = (eigen_[0].first)->dim();
        size_t eigen_size = eigen_.size();
        SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
        for (int i = 0; i < eigen_size; ++i) {
            evecs->set_column(0, i, (eigen_[i]).first);
        }

        // CI_RDMS for the targeted root
        CI_RDMS ci_rdms(options_, fci_ints_, determinant_, evecs, root_, root_);
        ci_rdms.set_symmetry(root_sym_);

        if (dm0_sum > 1.0e-12) {
            // compute RDMS and put into BlockedTensor format
            ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
            ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

            // loop over directions
            for (int z = 0; z < 3; ++z) {
                if (do_dm[z]) {
                    dm[z] = relaxed_dm_helper(dm0[z], dm1[z], dm2[z], D1, D2);
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
            SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
            for (int i = 0; i < eigen_size; ++i) {
                evecs->set_column(0, i, (eigens_[n][i]).first);
            }

            // loop over nroots for current symmetry
            for (int i = 0; i < nroots; ++i) {
                std::string name = generate_name(multi, i, irrep);
                std::vector<double> dm(3, 0.0);

                CI_RDMS ci_rdms(options_, fci_ints_, p_spaces_[n], evecs, i, i);
                ci_rdms.set_symmetry(irrep);

                if (dm0_sum > 1.0e-12) {
                    // compute RDMS and put into BlockedTensor format
                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                    // loop over directions
                    for (int z = 0; z < 3; ++z) {
                        if (do_dm[z]) {
                            dm[z] = relaxed_dm_helper(dm0[z], dm1[z], dm2[z], D1, D2);
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
FCI_MO::compute_relaxed_osc(std::vector<BlockedTensor>& dm1, std::vector<BlockedTensor>& dm2) {
    std::map<std::string, std::vector<double>> out;

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> mt{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    std::string pg = ct.symbol();
    int width = 2;
    if (pg == "cs" || pg == "d2h") {
        width = 3;
    } else if (pg == "c1") {
        width = 1;
    }
    auto generate_name = [&](const int& multi0, const int& root0, const int& irrep0,
                             const int& multi1, const int& root1, const int& irrep1) {
        std::string symbol0 = ct.gamma(irrep0).symbol();
        std::string symbol1 = ct.gamma(irrep1).symbol();
        std::stringstream name_ss;
        name_ss << std::setw(2) << root0 << " " << std::setw(7) << mt[multi0 - 1] << " "
                << std::setw(width) << symbol0 << " -> " << std::setw(2) << root1 << " "
                << std::setw(7) << mt[multi1 - 1] << " " << std::setw(width) << symbol1;
        return name_ss.str();
    };

    int nentry = sa_info_.size();
    for (int A = 0; A < nentry; ++A) {
        int irrep0, multi0, nroots0;
        std::vector<double> weights0;
        std::tie(irrep0, multi0, nroots0, weights0) = sa_info_[A];

        int ndets0 = (eigens_[A][0].first)->dim();
        SharedMatrix evecs0(new Matrix("evecs", ndets0, nroots0));
        for (int i = 0; i < nroots0; ++i) {
            evecs0->set_column(0, i, (eigens_[A][i]).first);
        }

        // oscillator strength of the same symmetry
        for (int i = 0; i < nroots0; ++i) {
            for (int j = i + 1; j < nroots0; ++j) {
                std::string name = generate_name(multi0, i, irrep0, multi0, j, irrep0);

                double Eex = eigens_[A][j].second - eigens_[A][i].second;
                std::vector<double> osc(3, 0.0);

                CI_RDMS ci_rdms(options_, fci_ints_, p_spaces_[A], evecs0, i, j);

                ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                for (int z = 0; z < 3; ++z) {
                    double dm = relaxed_dm_helper(0.0, dm1[z], dm2[z], D1, D2);
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
            int ndets1 = (eigens_[B][0].first)->dim();
            int ndets = ndets0 + ndets1;
            int nroots = nroots0 + nroots1;
            SharedMatrix evecs(new Matrix("evecs", ndets, nroots));

            for (int n = 0; n < nroots0; ++n) {
                SharedVector evec0 = evecs0->get_column(0, n);
                SharedVector evec(new Vector("combined evec0 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets0; ++i) {
                    evec->set(i, evec0->get(i));
                }
                evecs->set_column(0, n, evec);
            }

            for (int n = 0; n < nroots1; ++n) {
                SharedVector evec1 = eigens_[B][n].first;
                SharedVector evec(new Vector("combined evec1 " + std::to_string(n), ndets));
                for (size_t i = 0; i < ndets1; ++i) {
                    evec->set(i + ndets0, evec1->get(i));
                }
                evecs->set_column(0, n + nroots0, evec);
            }

            // combine p_space
            std::vector<STLBitsetDeterminant> p_space(p_spaces_[A]);
            std::vector<STLBitsetDeterminant>& p_space1 = p_spaces_[B];
            p_space.insert(p_space.end(), p_space1.begin(), p_space1.end());

            for (int i = 0; i < nroots0; ++i) {
                for (int j = 0; j < nroots1; ++j) {
                    std::string name = generate_name(multi0, i, irrep0, multi1, j, irrep1);

                    double Eex = eigens_[B][j].second - eigens_[A][i].second;
                    std::vector<double> osc(3, 0.0);

                    CI_RDMS ci_rdms(options_, fci_ints_, p_space, evecs, i, j + nroots0);

                    ambit::BlockedTensor D1 = compute_n_rdm(ci_rdms, 1);
                    ambit::BlockedTensor D2 = compute_n_rdm(ci_rdms, 2);

                    for (int z = 0; z < 3; ++z) {
                        double dm = relaxed_dm_helper(0.0, dm1[z], dm2[z], D1, D2);
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
        throw PSIEXCEPTION("Cannot compute RDMs except 1, 2, 3.");
    }

    ambit::BlockedTensor out;
    if (order == 1) {
        std::vector<double> opdm_a, opdm_b;
        cirdm.compute_1rdm(opdm_a, opdm_b);

        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D1", spin_cases({"aa"}));
        out.block("aa").data() = std::move(opdm_a);
        out.block("AA").data() = std::move(opdm_b);
    } else if (order == 2) {
        std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
        cirdm.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D2", spin_cases({"aaaa"}));
        out.block("aaaa").data() = std::move(tpdm_aa);
        out.block("aAaA").data() = std::move(tpdm_ab);
        out.block("AAAA").data() = std::move(tpdm_bb);
    } else if (order == 3) {
        std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;
        cirdm.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

        out = ambit::BlockedTensor::build(ambit::CoreTensor, "D3", spin_cases({"aaaaaa"}));
        out.block("aaaaaa").data() = std::move(tpdm_aaa);
        out.block("aaAaaA").data() = std::move(tpdm_aab);
        out.block("aAAaAA").data() = std::move(tpdm_abb);
        out.block("AAAAAA").data() = std::move(tpdm_bbb);
    }

    return out;
}

double FCI_MO::relaxed_dm_helper(const double& dm0, BlockedTensor& dm1, BlockedTensor& dm2,
                                 BlockedTensor& D1, BlockedTensor& D2) {
    double dm_out = dm0;

    dm_out += dm1["uv"] * D1["uv"];
    dm_out += dm1["UV"] * D1["UV"];
    dm_out += 0.25 * dm2["uvxy"] * D2["uvxy"];
    dm_out += 0.25 * dm2["UVXY"] * D2["UVXY"];
    dm_out += dm2["uVxY"] * D2["uVxY"];

    return dm_out;
}

d3 FCI_MO::compute_orbital_extents() {

    // compute AO quadrupole integrals
    std::shared_ptr<BasisSet> basisset = this->basisset();
    std::shared_ptr<IntegralFactory> ints = std::shared_ptr<IntegralFactory>(
        new IntegralFactory(basisset, basisset, basisset, basisset));

    std::vector<SharedMatrix> ao_Qpole;
    for (const std::string& direction : {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}) {
        std::string name = "AO Quadrupole" + direction;
        ao_Qpole.push_back(SharedMatrix(new Matrix(name, basisset->nbf(), basisset->nbf())));
    }
    std::shared_ptr<OneBodyAOInt> aoqOBI(ints->ao_quadrupole());
    aoqOBI->compute(ao_Qpole);

    // orbital coefficients arranged by orbital energies
    SharedMatrix Ca_ao = this->Ca_subset("AO");
    int nao = Ca_ao->nrow();
    int nmo = Ca_ao->ncol();

    std::vector<SharedVector> quadrupole;
    quadrupole.push_back(SharedVector(new Vector("Orbital Quadrupole XX", nmo)));
    quadrupole.push_back(SharedVector(new Vector("Orbital Quadrupole YY", nmo)));
    quadrupole.push_back(SharedVector(new Vector("Orbital Quadrupole ZZ", nmo)));

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

    SharedVector epsilon_a = this->epsilon_a();
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
        size_t na = active_[h];
        if (na == 0)
            continue;
        orb_extents[h] = d2(na, d1());
    }

    for (int n = 0, size = metric.size(); n < size; ++n) {
        double epsilon;
        int i, h;
        std::tie(epsilon, i, h) = metric[n];

        int offset = frzcpi_[h] + core_[h];
        if (i < offset || i >= offset + active_[h])
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
            for (size_t i = 0; i < active_[h]; ++i) {
                double orbext = orb_extents[h][i][0] + orb_extents[h][i][1] + orb_extents[h][i][2];

                if (orbext > 1.0e6) {
                    diffused_orbs_.push_back(i + offset);

                    if (h == 0) {
                        idx_diffused_ = i; // totally symmetric diffused orbital
                        found = true;
                    }
                }
            }
            offset += active_[h];
        }

        if (!found) {
            outfile->Printf("\n  Totally symmetric diffused orbital is not found.");
            outfile->Printf("\n  Make sure a diffused s function is added to the basis.");
            throw PSIEXCEPTION("Totally symmetric diffused orbital is not found.");
        }
    }

    return orb_extents;
}

void FCI_MO::fill_density() {
    L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {na_, na_});
    L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {na_, na_});
    L1a.iterate([&](const ::vector<size_t>& i, double& value) {
        size_t np = idx_a_[i[0]];
        size_t nq = idx_a_[i[1]];
        value = Da_[np][nq];
    });
    L1b.iterate([&](const ::vector<size_t>& i, double& value) {
        size_t np = idx_a_[i[0]];
        size_t nq = idx_a_[i[1]];
        value = Db_[np][nq];
    });
}

// void FCI_MO::fill_cumulant2() {
//    L2aa.iterate(
//        [&](const ::vector<size_t>& i, double& value) { value = L2aa_[i[0]][i[1]][i[2]][i[3]]; });
//    L2ab.iterate(
//        [&](const ::vector<size_t>& i, double& value) { value = L2ab_[i[0]][i[1]][i[2]][i[3]]; });
//    L2bb.iterate(
//        [&](const ::vector<size_t>& i, double& value) { value = L2bb_[i[0]][i[1]][i[2]][i[3]]; });
//}

// void FCI_MO::fill_cumulant3() {
//    L3aaa.iterate([&](const ::vector<size_t>& i, double& value) {
//        value = L3aaa_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
//    });
//    L3aab.iterate([&](const ::vector<size_t>& i, double& value) {
//        value = L3aab_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
//    });
//    L3abb.iterate([&](const ::vector<size_t>& i, double& value) {
//        value = L3abb_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
//    });
//    L3bbb.iterate([&](const ::vector<size_t>& i, double& value) {
//        value = L3bbb_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
//    });
//}

void FCI_MO::fill_density(vector<double>& opdm_a, std::vector<double>& opdm_b) {
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));
    L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {na_, na_});
    L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {na_, na_});

    // fill in L1a and L1b
    L1a.data() = opdm_a;
    L1b.data() = opdm_b;

    // fill in Da_ and Db_
    for (size_t p = 0; p < nc_; ++p) {
        size_t np = idx_c_[p];
        Da_[np][np] = 1.0;
        Db_[np][np] = 1.0;
    }

    for (size_t p = 0; p < na_; ++p) {
        size_t np = idx_a_[p];
        for (size_t q = p; q < na_; ++q) {
            size_t nq = idx_a_[q];

            if ((sym_active_[p] ^ sym_active_[q]) != 0)
                continue;

            size_t index = p * na_ + q;
            Da_[np][nq] = opdm_a[index];
            Db_[np][nq] = opdm_b[index];

            Da_[nq][np] = Da_[np][nq];
            Db_[nq][np] = Db_[np][nq];
        }
    }
}

void FCI_MO::compute_ref(const int& level) {
    timer_on("Compute Ref");
    Timer tcu;
    if (!quiet_) {
        std::string name = "1-";
        if (level >= 2) {
            name = "1- and 2-";
            if (level >= 3) {
                name = "1-, 2- and 3-";
            }
        }
        outfile->Printf("\n  Computing %scumulants ... ", name.c_str());
    }

    // prepare ci_rdms
    int dim = (eigen_[0].first)->dim();
    size_t eigen_size = eigen_.size();
    SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
    for (int i = 0; i < eigen_size; ++i) {
        evecs->set_column(0, i, (eigen_[i]).first);
    }
    CI_RDMS ci_rdms(options_, fci_ints_, determinant_, evecs, root_, root_);

    // compute 1-RDM
    std::vector<double> opdm_a, opdm_b;
    ci_rdms.compute_1rdm(opdm_a, opdm_b);

    // fill in L1a and L1b tensors
    fill_density(opdm_a, opdm_b);

    // compute 2-RDM
    if (level >= 2) {
        std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
        ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

        // fill in L2aa, L2ab, L2bb tensors
        compute_cumulant2(tpdm_aa, tpdm_ab, tpdm_bb);
    }

    // compute 3-RDM
    string threepdc = options_.get_str("THREEPDC");
    if (threepdc != "ZERO" && level >= 3) {
        std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;
        if (threepdc == "MK") {
            ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);
        } else if (threepdc == "MK_DECOMP") {
            size_t nelement3 = na_ * na_ * na_ * na_ * na_ * na_;
            tpdm_aaa = std::vector<double>(nelement3, 0.0);
            tpdm_aab = std::vector<double>(nelement3, 0.0);
            tpdm_abb = std::vector<double>(nelement3, 0.0);
            tpdm_bbb = std::vector<double>(nelement3, 0.0);
        }

        // fill in L3aaa, L3aab, L3abb, L3bbb tensors
        compute_cumulant3(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);
    }

    if (!quiet_) {
        outfile->Printf("Done. Timing %15.6f s\n", tcu.get());
    }
    timer_off("Compute Ref");
}

Reference FCI_MO::reference(const int& level) {
    Reference ref;
    ref.set_Eref(Eref_);

    if (options_["AVG_STATE"].size() != 0) {
        compute_sa_ref(level);
    } else {
        compute_ref(level);
    }

    if (level > 0) {
        ref.set_L1a(L1a);
        ref.set_L1b(L1b);
    }

    if (level > 1) {
        ref.set_L2aa(L2aa);
        ref.set_L2ab(L2ab);
        ref.set_L2bb(L2bb);
    }

    if (level > 2 && (options_.get_str("THREEPDC") != "ZERO")) {
        ref.set_L3aaa(L3aaa);
        ref.set_L3aab(L3aab);
        ref.set_L3abb(L3abb);
        ref.set_L3bbb(L3bbb);
    }
    return ref;
}

void FCI_MO::fill_naive_cumulants(Reference& ref, const int& level) {
    // fill in 1-cumulant (same as 1-RDM) to D1a_, D1b_
    ambit::Tensor L1a = ref.L1a();
    ambit::Tensor L1b = ref.L1b();
    fill_one_cumulant(L1a, L1b);
    if (print_ > 1) {
        print_density("Alpha", Da_);
        print_density("Beta", Db_);
    }

    // fill in 2-cumulant to L2aa_, L2ab_, L2bb_
    if (level >= 2) {
        ambit::Tensor L2aa = ref.L2aa();
        ambit::Tensor L2ab = ref.L2ab();
        ambit::Tensor L2bb = ref.L2bb();
        fill_two_cumulant(L2aa, L2ab, L2bb);
        if (print_ > 2) {
            print2PDC("L2aa", L2aa_, print_);
            print2PDC("L2ab", L2ab_, print_);
            print2PDC("L2bb", L2bb_, print_);
        }
    }

    // fill in 3-cumulant to L3aaa_, L3aab_, L3abb_, L3bbb_
    if (level >= 3) {
        ambit::Tensor L3aaa = ref.L3aaa();
        ambit::Tensor L3aab = ref.L3aab();
        ambit::Tensor L3abb = ref.L3abb();
        ambit::Tensor L3bbb = ref.L3bbb();
        fill_three_cumulant(L3aaa, L3aab, L3abb, L3bbb);
        if (print_ > 3) {
            print3PDC("L3aaa", L3aaa_, print_);
            print3PDC("L3aab", L3aab_, print_);
            print3PDC("L3abb", L3abb_, print_);
            print3PDC("L3bbb", L3bbb_, print_);
        }
    }
}

void FCI_MO::fill_one_cumulant(ambit::Tensor& L1a, ambit::Tensor& L1b) {
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));

    for (size_t p = 0; p < nc_; ++p) {
        size_t np = idx_c_[p];
        Da_[np][np] = 1.0;
        Db_[np][np] = 1.0;
    }

    std::vector<double>& opdc_a = L1a.data();
    std::vector<double>& opdc_b = L1b.data();

    // TODO: try omp here
    for (size_t p = 0; p < na_; ++p) {
        size_t np = idx_a_[p];
        for (size_t q = p; q < na_; ++q) {
            size_t nq = idx_a_[q];

            if ((sym_active_[p] ^ sym_active_[q]) != 0)
                continue;

            size_t index = p * na_ + q;
            Da_[np][nq] = opdc_a[index];
            Db_[np][nq] = opdc_b[index];

            Da_[nq][np] = Da_[np][nq];
            Db_[nq][np] = Db_[np][nq];
        }
    }
}

void FCI_MO::fill_two_cumulant(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb) {
    L2aa_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2ab_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2bb_ = d4(na_, d3(na_, d2(na_, d1(na_))));

    std::vector<double>& tpdc_aa = L2aa.data();
    std::vector<double>& tpdc_ab = L2ab.data();
    std::vector<double>& tpdc_bb = L2bb.data();

    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;

    // TODO: try omp here
    for (size_t p = 0; p < na_; ++p) {
        for (size_t q = 0; q < na_; ++q) {
            for (size_t r = 0; r < na_; ++r) {
                for (size_t s = 0; s < na_; ++s) {

                    if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0)
                        continue;

                    size_t index = p * dim3 + q * dim2 + r * na_ + s;

                    L2aa_[p][q][r][s] = tpdc_aa[index];
                    L2ab_[p][q][r][s] = tpdc_ab[index];
                    L2bb_[p][q][r][s] = tpdc_bb[index];
                }
            }
        }
    }
}

void FCI_MO::fill_three_cumulant(ambit::Tensor& L3aaa, ambit::Tensor& L3aab, ambit::Tensor& L3abb,
                                 ambit::Tensor& L3bbb) {
    L3aaa_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3aab_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3abb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3bbb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));

    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;
    size_t dim4 = na_ * dim3;
    size_t dim5 = na_ * dim4;

    auto fill = [&](d6& L3, ambit::Tensor& L3t) {
        std::vector<double>& data = L3t.data();

        // TODO: try omp here
        for (size_t p = 0; p != na_; ++p) {
            for (size_t q = 0; q != na_; ++q) {
                for (size_t r = 0; r != na_; ++r) {
                    for (size_t s = 0; s != na_; ++s) {
                        for (size_t t = 0; t != na_; ++t) {
                            for (size_t u = 0; u != na_; ++u) {

                                if ((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^
                                     sym_active_[s] ^ sym_active_[t] ^ sym_active_[u]) != 0)
                                    continue;

                                size_t index =
                                    p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * na_ + u;

                                L3[p][q][r][s][t][u] = data[index];
                            }
                        }
                    }
                }
            }
        }
    };

    fill(L3aaa_, L3aaa);
    fill(L3aab_, L3aab);
    fill(L3abb_, L3abb);
    fill(L3bbb_, L3bbb);
}

// void FCI_MO::set_orbs(SharedMatrix Ca, SharedMatrix Cb) {
//    SharedMatrix Ca_wfn = this->Ca();
//    SharedMatrix Cb_wfn = this->Cb();
//    Ca_wfn->copy(Ca);
//    Cb_wfn->copy(Cb);
//    integral_->retransform_integrals();
//    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
//    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
//    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
//    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
//    fci_ints_->compute_restricted_one_body_operator();
//}

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
        size_t eigen_size = eigen_.size();
        if (nroot_ > eigen_size) {
            outfile->Printf("\n  Too many roots of interest!");
            std::string be = (eigen_size > 1) ? "are" : "is";
            std::string plural = (eigen_size > 1) ? "roots" : "root";
            outfile->Printf("\n  There %s only %3d %s that satisfy the condition!", be.c_str(),
                            eigen_size, plural.c_str());
            outfile->Printf("\n  Check root_sym, multi, and number of determinants.");
            throw PSIEXCEPTION("Too many roots of interest.");
        }
        print_CI(nroot_, options_.get_double("FCIMO_PRINT_CIVEC"), eigen_, determinant_);

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

    //    // rotate references such that <M|F|N> is diagonal
    //    int nentry = eigens_.size();
    //    for(int i = 0; i < nentry; ++i){ // TODO: rotation should not do this,
    //    currently transition density should be in the same irrep
    //        int irrep = options_["AVG_STATE"][i][0].to_integer();

    //        auto& eigen = eigens_[i];
    //        int nroots = eigen.size();
    //        SharedMatrix Fock_MN (new Matrix("Fock_MN",nroots,nroots));

    //        int dim = (eigen[0].first)->dim();
    //        SharedMatrix evecs (new Matrix("evecs",dim,dim));
    //        for(int M = 0; M < nroots; ++M){
    //            evecs->set_column(0,M,(eigen[M]).first);
    //        }

    //        // compute matrix elements of Fock_MN
    //        for(int M = 0; M < nroots; ++M){
    //            for(int N = M; N < nroots; ++N){

    //                // compute density
    //                CI_RDMS ci_rdms
    //                (options_,fci_ints_,determinant_,evecs,M,N);
    //                ci_rdms.set_symmetry(irrep);
    //                std::vector<double> opdm_a (nelement, 0.0);
    //                std::vector<double> opdm_b (nelement, 0.0);
    //                ci_rdms.compute_1rdm(opdm_a,opdm_b);

    //                // contract density with SA-Fock
    //                double fuv = 0.0;
    //                for(size_t u = 0; u < na_; ++u){
    //                    size_t nu = idx_a_[u];

    //                    // just in case SA-Fock is not diaognal
    //                    for(size_t v = 0; v < na_; ++v){
    //                        size_t nv = idx_a_[v];

    //                        fuv += Fa_[nu][nv] * opdm_a[v * na_ + u];
    //                        fuv += Fb_[nu][nv] * opdm_b[v * na_ + u];
    //                    }
    //                }

    //                if(M == N){
    //                    Fock_MN->set(M, M, fuv);
    //                }else{
    //                    Fock_MN->set(M, N, fuv);
    //                    Fock_MN->set(N, M, fuv);
    //                }

    //            }
    //        }

    //        // diaognalize Fock_MN
    ////        Fock_MN->print();
    //        SharedMatrix Fvecs (new Matrix("Fock_MN evecs", nroots, nroots));
    //        SharedVector Fvals (new Vector("Fock_MN evals", nroots));
    //        Fock_MN->diagonalize(Fvecs, Fvals);
    ////        Fvecs->eivprint(Fvals);

    //        // rotate eigen vectors
    //        evecs = SharedMatrix (new Matrix("evecs", dim, nroots));
    //        for(int M = 0; M < nroots; ++M){
    //            evecs->set_column(0,M,(eigen[M]).first);
    //        }
    //        SharedMatrix rvecs (new Matrix("Rotated evecs", dim, nroots));
    //        rvecs->gemm(false,false,1.0,evecs,Fvecs,0.0);

    //        for(int M = 0; M < nroots; ++M){
    //            (eigens_[i][M].first)->print();
    //        }
    //        for(int M = 0; M < nroots; ++M){
    //            eigen[M].first = rvecs->get_column(0,M);
    //        }
    //        for(int M = 0; M < nroots; ++M){
    //            (eigens_[i][M].first)->print();
    //        }
    //    }

    Eref_ = Ecas_sa;
    Process::environment.globals["CURRENT ENERGY"] = Ecas_sa;
    return Ecas_sa;
}

// TODO this function probably should not be here.
void FCI_MO::xms_rotate(const int& irrep) {
    int nentry = eigens_.size();
    for (int i = 0; i < nentry; ++i) {
        int this_irrep = options_["AVG_STATE"][i][0].to_integer();
        if (this_irrep != irrep) {
            continue;
        } else {
            size_t nelement = na_ * na_;
            auto& eigen = eigens_[i];
            int nroots = eigen.size();
            SharedMatrix Fock_MN(new Matrix("Fock_MN", nroots, nroots));

            int dim = (eigen[0].first)->dim();
            SharedMatrix evecs(new Matrix("evecs", dim, nroots));
            for (int M = 0; M < nroots; ++M) {
                evecs->set_column(0, M, (eigen[M]).first);
            }

            // compute matrix elements of Fock_MN
            for (int M = 0; M < nroots; ++M) {
                for (int N = M; N < nroots; ++N) {

                    // compute density
                    CI_RDMS ci_rdms(options_, fci_ints_, determinant_, evecs, M, N);
                    ci_rdms.set_symmetry(irrep);
                    std::vector<double> opdm_a(nelement, 0.0);
                    std::vector<double> opdm_b(nelement, 0.0);
                    ci_rdms.compute_1rdm(opdm_a, opdm_b);

                    // contract density with SA-Fock
                    double fuv = 0.0;
                    for (size_t u = 0; u < na_; ++u) {
                        size_t nu = idx_a_[u];

                        for (size_t v = 0; v < na_; ++v) {
                            size_t nv = idx_a_[v];

                            fuv += Fa_[nu][nv] * opdm_a[v * na_ + u];
                            fuv += Fb_[nu][nv] * opdm_b[v * na_ + u];
                        }
                    }

                    if (M == N) {
                        Fock_MN->set(M, M, fuv); // ignored core part (a constant shift)
                    } else {
                        Fock_MN->set(M, N, fuv);
                        Fock_MN->set(N, M, fuv);
                    }
                }
            }

            // diaognalize Fock_MN
            SharedMatrix Fvecs(new Matrix("Fock_MN evecs", nroots, nroots));
            SharedVector Fvals(new Vector("Fock_MN evals", nroots));
            Fock_MN->diagonalize(Fvecs, Fvals);
            //            Fvecs->eivprint(Fvals);

            // rotate eigen vectors
            evecs = SharedMatrix(new Matrix("evecs", dim, nroots));
            for (int M = 0; M < nroots; ++M) {
                evecs->set_column(0, M, (eigen[M]).first);
            }
            SharedMatrix rvecs(new Matrix("Rotated evecs", dim, nroots));
            rvecs->gemm(false, false, 1.0, evecs, Fvecs, 0.0);

            //            // recompute reference energies
            //            ambit::Tensor tei_active_aa =
            //            integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_,
            //            idx_a_);
            //            ambit::Tensor tei_active_ab =
            //            integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_,
            //            idx_a_);
            //            ambit::Tensor tei_active_bb =
            //            integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_,
            //            idx_a_);
            //            fci_ints_->
        }
    }
}

void FCI_MO::compute_sa_ref(const int& level) {
    timer_on("Compute SA Ref");
    Timer tcu;
    if (!quiet_) {
        std::string name = "1-";
        if (level >= 2) {
            name = "1- and 2-";
            if (level >= 3) {
                name = "1-, 2- and 3-";
            }
        }
        outfile->Printf("\n  Computing %scumulants ... ", name.c_str());
    }

    // prepare averaged densities
    size_t nelement = na_ * na_;
    std::vector<double> sa_opdm_a(nelement, 0.0);
    std::vector<double> sa_opdm_b(nelement, 0.0);

    size_t nelement2 = nelement * nelement;
    std::vector<double> sa_tpdm_aa, sa_tpdm_ab, sa_tpdm_bb;
    if (level >= 2) {
        sa_tpdm_aa = std::vector<double>(nelement2, 0.0);
        sa_tpdm_ab = std::vector<double>(nelement2, 0.0);
        sa_tpdm_bb = std::vector<double>(nelement2, 0.0);
    }

    size_t nelement3 = nelement * nelement2;
    std::vector<double> sa_tpdm_aaa, sa_tpdm_aab, sa_tpdm_abb, sa_tpdm_bbb;
    bool no_3pdc = (options_.get_str("THREEPDC") == "ZERO");
    if (level >= 3 && (!no_3pdc)) {
        sa_tpdm_aaa = std::vector<double>(nelement3, 0.0);
        sa_tpdm_aab = std::vector<double>(nelement3, 0.0);
        sa_tpdm_abb = std::vector<double>(nelement3, 0.0);
        sa_tpdm_bbb = std::vector<double>(nelement3, 0.0);
    }

    // function that scale pdm by w and add scaled pdm to sa_pdm
    auto scale_add = [](std::vector<double>& sa_pdm, std::vector<double>& pdm, const double& w) {
        //        std::transform(pdm.begin(), pdm.end(), pdm.begin(),
        //                       std::bind1st(std::multiplies<double>(), w));
        std::for_each(pdm.begin(), pdm.end(), [&](double& v) { v *= w; });
        std::transform(sa_pdm.begin(), sa_pdm.end(), pdm.begin(), sa_pdm.begin(),
                       std::plus<double>());
    };

    // loop over all averaged states
    int nentry = sa_info_.size();
    for (int n = 0; n < nentry; ++n) {
        // get current nroots and weights
        int nroots, irrep;
        std::vector<double> weights;
        std::tie(irrep, std::ignore, nroots, weights) = sa_info_[n];

        // prepare eigen vectors for current symmetry
        int dim = (eigens_[n][0].first)->dim();
        size_t eigen_size = eigens_[n].size();
        SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
        for (int i = 0; i < eigen_size; ++i) {
            evecs->set_column(0, i, (eigens_[n][i]).first);
        }

        for (int i = 0; i < nroots; ++i) {
            double weight = weights[i];
            CI_RDMS ci_rdms(options_, fci_ints_, p_spaces_[n], evecs, i, i);
            ci_rdms.set_symmetry(irrep);

            // compute 1-RDMs
            std::vector<double> opdm_a, opdm_b;
            ci_rdms.compute_1rdm(opdm_a, opdm_b);

            scale_add(sa_opdm_a, opdm_a, weight);
            scale_add(sa_opdm_b, opdm_b, weight);

            // compute 2-RDMs
            if (level >= 2) {
                std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
                ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

                scale_add(sa_tpdm_aa, tpdm_aa, weight);
                scale_add(sa_tpdm_ab, tpdm_ab, weight);
                scale_add(sa_tpdm_bb, tpdm_bb, weight);
            }

            if (level >= 3 && (!no_3pdc)) {
                std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;
                ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

                scale_add(sa_tpdm_aaa, tpdm_aaa, weight);
                scale_add(sa_tpdm_aab, tpdm_aab, weight);
                scale_add(sa_tpdm_abb, tpdm_abb, weight);
                scale_add(sa_tpdm_bbb, tpdm_bbb, weight);
            }
        }

    } // end looping over all averaged states

    // fill in L1 tensors
    fill_density(sa_opdm_a, sa_opdm_b);

    // compute 2-cumulants and fill in L2 tensors
    if (level >= 2) {
        compute_cumulant2(sa_tpdm_aa, sa_tpdm_ab, sa_tpdm_bb);
    }

    // compute 3-cumulants and fill in L3 tensors
    if (level >= 3 && (!no_3pdc)) {
        compute_cumulant3(sa_tpdm_aaa, sa_tpdm_aab, sa_tpdm_abb, sa_tpdm_bbb);
    }

    if (!quiet_) {
        outfile->Printf("Done. Timing %15.6f s\n", tcu.get());
    }
    timer_off("Compute SA Ref");
}

void FCI_MO::compute_cumulant2(vector<double>& tpdm_aa, std::vector<double>& tpdm_ab,
                               std::vector<double>& tpdm_bb) {
    L2aa = ambit::Tensor::build(ambit::CoreTensor, "L2aa", {na_, na_, na_, na_});
    L2ab = ambit::Tensor::build(ambit::CoreTensor, "L2ab", {na_, na_, na_, na_});
    L2bb = ambit::Tensor::build(ambit::CoreTensor, "L2bb", {na_, na_, na_, na_});

    // copy incoming 2rdms to 2cumulants
    L2aa.data() = std::move(tpdm_aa);
    L2ab.data() = std::move(tpdm_ab);
    L2bb.data() = std::move(tpdm_bb);

    // add wedge product of 1cumulants (1rdms)
    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");
}

void FCI_MO::compute_cumulant3(vector<double>& tpdm_aaa, std::vector<double>& tpdm_aab,
                               std::vector<double>& tpdm_abb, std::vector<double>& tpdm_bbb) {
    // aaa
    L3aaa = ambit::Tensor::build(ambit::CoreTensor, "L3aaa", {na_, na_, na_, na_, na_, na_});
    L3aaa.data() = std::move(tpdm_aaa);

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
    L3aab = ambit::Tensor::build(ambit::CoreTensor, "L3aab", {na_, na_, na_, na_, na_, na_});
    L3aab.data() = std::move(tpdm_aab);

    L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
    L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

    // abb
    L3abb = ambit::Tensor::build(ambit::CoreTensor, "L3abb", {na_, na_, na_, na_, na_, na_});
    L3abb.data() = std::move(tpdm_abb);

    L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
    L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

    // bbb
    L3bbb = ambit::Tensor::build(ambit::CoreTensor, "L3bbb", {na_, na_, na_, na_, na_, na_});
    L3bbb.data() = std::move(tpdm_bbb);

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
}
}
}
