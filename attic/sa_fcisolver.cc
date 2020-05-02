/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <iomanip>
#include <sstream>
#include <tuple>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"

#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"
#include "helpers/helpers.h"
#include "sa_fcisolver.h"

using namespace psi;

namespace forte {

SA_FCISolver::SA_FCISolver(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> wfn)
    : options_(options), wfn_(wfn) {
    read_options();
}

void SA_FCISolver::read_options() {
    // irrep symbol
    int nirrep = wfn_->nirrep();
    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0; h < nirrep; ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // read states for averaging
    nstates_ = 0;
    parsed_options_.clear();

    if (options_["AVG_STATE"].has_changed()) {
        size_t nentry = options_["AVG_STATE"].size();

        // figure out total number of states
        std::vector<int> nstatespim;
        std::vector<int> irreps;
        std::vector<int> multis;
        for (size_t i = 0; i < nentry; ++i) {
            if (options_["AVG_STATE"][i].size() != 3) {
                outfile->Printf("\n  Error: invalid input of AVG_STATE. Each "
                                "entry should take an array of three numbers.");
                throw psi::PSIEXCEPTION("Invalid input of AVG_STATE");
            }

            // irrep
            int irrep = options_["AVG_STATE"][i][0].to_integer();
            if (irrep >= nirrep || irrep < 0) {
                outfile->Printf("\n  Error: invalid irrep in AVG_STATE. Please "
                                "check the input irrep (start from 0) not to "
                                "exceed %d",
                                nirrep - 1);
                throw psi::PSIEXCEPTION("Invalid irrep in AVG_STATE");
            }
            irreps.push_back(irrep);

            // multiplicity
            int multi = options_["AVG_STATE"][i][1].to_integer();
            if (multi < 1) {
                outfile->Printf("\n  Error: invalid multiplicity in AVG_STATE.");
                throw psi::PSIEXCEPTION("Invaid multiplicity in AVG_STATE");
            }
            multis.push_back(multi);

            // number of states of irrep and multiplicity
            int nstates_this = options_["AVG_STATE"][i][2].to_integer();
            if (nstates_this < 1) {
                outfile->Printf("\n  Error: invalid nstates in AVG_STATE. "
                                "nstates of a certain irrep and multiplicity "
                                "should greater than 0.");
                throw psi::PSIEXCEPTION("Invalid nstates in AVG_STATE.");
            }
            nstatespim.push_back(nstates_this);
            nstates_ += nstates_this;
        }

        // test input weights
        std::vector<std::vector<double>> weights;
        if (options_["AVG_WEIGHT"].has_changed()) {
            if (options_["AVG_WEIGHT"].size() != nentry) {
                outfile->Printf("\n  Error: mismatched number of entries in "
                                "AVG_STATE (%d) and AVG_WEIGHT (%d).",
                                nentry, options_["AVG_WEIGHT"].size());
                throw psi::PSIEXCEPTION("Mismatched number of entries in AVG_STATE "
                                        "and AVG_WEIGHT.");
            }

            double wsum = 0.0;
            for (size_t i = 0; i < nentry; ++i) {
                int nw = options_["AVG_WEIGHT"][i].size();
                if (nw != nstatespim[i]) {
                    outfile->Printf("\n  Error: mismatched number of weights "
                                    "in entry %d of AVG_WEIGHT. Asked for %d "
                                    "states but only %d weights.",
                                    i, nstatespim[i], nw);
                    throw psi::PSIEXCEPTION("Mismatched number of weights in AVG_WEIGHT.");
                }

                std::vector<double> weight;
                for (int n = 0; n < nw; ++n) {
                    double w = options_["AVG_WEIGHT"][i][n].to_double();
                    if (w < 0.0) {
                        outfile->Printf("\n  Error: negative weights in AVG_WEIGHT.");
                        throw psi::PSIEXCEPTION("Negative weights in AVG_WEIGHT.");
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
                throw psi::PSIEXCEPTION("AVG_WEIGHT entries do not add up to 1.0.");
            }

        } else {
            // use equal weights
            double w = 1.0 / nstates_;
            for (size_t i = 0; i < nentry; ++i) {
                std::vector<double> weight(nstatespim[i], w);
                weights.push_back(weight);
            }
        }

        // form option parser
        for (size_t i = 0; i < nentry; ++i) {
            std::tuple<int, int, int, std::vector<double>> avg_info =
                std::make_tuple(irreps[i], multis[i], nstatespim[i], weights[i]);
            parsed_options_.push_back(avg_info);
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
        for (size_t i = 0; i < nentry; ++i) {
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
        outfile->Printf("\n    Total number of states: %d", nstates_);
        outfile->Printf("\n    %s", dash.c_str());
    }
}

double SA_FCISolver::compute_energy() {
    double E_sa_casscf = 0.0;
    Reference sa_cas;
    std::vector<double> casscf_energies;
    std::vector<Reference> sa_cas_ref;
    size_t na = mo_space_info_->size("ACTIVE");

    for (const auto& cas_solutions : parsed_options_) {
        int symmetry;
        int multiplicity;
        int nroot;
        std::tie(symmetry, multiplicity, nroot, std::ignore) = cas_solutions;

        psi::Dimension active_dim = mo_space_info_->dimension("ACTIVE");
        std::vector<size_t> rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
        std::vector<size_t> active = mo_space_info_->corr_absolute_mo("ACTIVE");

        int charge = psi::Process::environment.molecule()->molecular_charge();
        if (options_["CHARGE"].has_changed()) {
            charge = options_.get_int("CHARGE");
        }

        int nel = 0;
        int natom = psi::Process::environment.molecule()->natom();
        for (int i = 0; i < natom; i++) {
            nel += static_cast<int>(psi::Process::environment.molecule()->Z(i));
        }
        // If the charge has changed, recompute the number of electrons
        // Or if you cannot find the number of electrons
        nel -= charge;

        // Default: lowest spin solution
        int twice_ms = (multiplicity + 1) % 2;

        if (options_["MS"].has_changed()) {
            twice_ms = std::round(2.0 * options_.get_double("MS"));
        }

        if (twice_ms < 0) {
            outfile->Printf("\n  Ms must be no less than 0.");
            outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", twice_ms, multiplicity);
            outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
            throw psi::PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
        }

        if (options_.get_int("PRINT")) {
            print_h2("FCI Solver Summary");
            outfile->Printf("\n  Number of electrons: %d", nel);
            outfile->Printf("\n  Charge: %d", charge);
            outfile->Printf("\n  Multiplicity: %d", multiplicity);
            outfile->Printf("\n  Davidson subspace max dim: %d",
                            options_.get_int("DL_SUBSPACE_PER_ROOT"));
            outfile->Printf("\n  Davidson subspace min dim: %d",
                            options_.get_int("DL_COLLAPSE_PER_ROOT"));
            if (twice_ms % 2 == 0) {
                outfile->Printf("\n  M_s: %d", twice_ms / 2);
            } else {
                outfile->Printf("\n  M_s: %d/2", twice_ms);
            }
        }

        if (((nel - twice_ms) % 2) != 0)
            throw psi::PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

        // Adjust the number of for frozen and restricted doubly occupied
        size_t nactel = nel;

        size_t na = (nactel + twice_ms) / 2;
        size_t nb = nactel - na;
        StateInfo state(na, nb, multiplicity, twice_ms, symmetry);
        // TODO use base class info
        auto as_ints =
            make_active_space_ints(mo_space_info_, ints_, "ACTIVE", {{"RESTRICTED_DOCC"}});
        FCISolver fcisolver(state, nroot, mo_space_info_, as_ints);

        fcisolver.set_options(std::make_shared<ForteOptions>(options_));
        fcisolver.set_max_rdm_level(2);
        fcisolver.set_test_rdms(options_.get_bool("FCI_TEST_RDMS"));
        fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
        fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
        fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));
        fcisolver.set_print_no(false);
        if (fci_ints_ == nullptr) {
            outfile->Printf("\n\n You need to set fci_ints");
            throw psi::PSIEXCEPTION("Set FCI INTS");
        } else {
            fcisolver.set_active_space_integrals(fci_ints_);
        }

        //        fcisolver.set_root(0);
        //        fcisolver.set_test_rdms(false);
        //        fcisolver.compute_energy();
        //        double Enuc =
        //        psi::Process::environment.molecule()->nuclear_repulsion_energy();
        //        psi::SharedMatrix vecs = fcisolver.eigen_vecs();
        //        psi::SharedVector vals = fcisolver.eigen_vals();
        //        for(int n = 0; n < nroot; ++n){
        //            // create new FCIVector pointers
        //            std::shared_ptr<FCIVector> fci_wfn = fcisolver.get_FCIWFN();
        //            fci_wfn->copy(vecs->get_column(0,n));
        //            fci_wfn->print();
        //            SA_C_.push_back(fci_wfn); // this line probably would not
        //            work

        //            // use the FCIVector in FCISolver to compute RDM and obtain
        //            the reference
        //            fci_wfn->compute_rdms(2);
        //            for(double x: fci_wfn->opdm_a()){
        //                outfile->Printf("\n  %20.15f",x);
        //            }
        //            sa_cas_ref.push_back(fcisolver.reference());

        //            // fill in energies
        //            double Ecas = vals->get(0,n) + Enuc;
        //            casscf_energies.push_back(Ecas);
        //        }

        psi::SharedVector evals;
        double Enuc = psi::Process::environment.molecule()->nuclear_repulsion_energy(
            wfn_->get_dipole_field_strength());

        std::vector<std::pair<size_t,size_t>> roots;
        for (int root_number = 0; root_number < nroot; root_number++) {
            roots.push_back(std::make_pair(root_number,root_number));
        }
        sa_cas_ref = fcisolver.get_reference(roots);
        
        for (int root_number = 0; root_number < nroot; root_number++) {
            fcisolver.set_root(root_number);

            // only diagonalize the Hamiltonian once
            // note: compute_energy of FCISolver computes energy (all roots) and RDMs (given root)
            if (root_number == 0) {
                fcisolver.compute_energy();
                evals = fcisolver.evals();
            } else {
                fcisolver.compute_rdms_root(root_number);
            }

            //            SA_C_.push_back(fcisolver.get_FCIWFN());
            double Ecasscf = evals->get(root_number) + Enuc;
            casscf_energies.push_back(Ecasscf);
            sa_cas_ref[root_number].set_Eref(Ecasscf);
        }
    }
    if (!options_["AVG_WEIGHT"].has_changed()) {
        for (auto& casscf_energy : casscf_energies) {
            E_sa_casscf += casscf_energy;
        }
        E_sa_casscf /= casscf_energies.size();
        sa_cas.set_Eref(E_sa_casscf);
        ambit::Tensor L1a_sa = ambit::Tensor::build(ambit::CoreTensor, "L1a_sa", {na, na});
        ambit::Tensor L1b_sa = ambit::Tensor::build(ambit::CoreTensor, "L1b_sa", {na, na});
        ambit::Tensor L2aa_sa =
            ambit::Tensor::build(ambit::CoreTensor, "L2aa_sa", {na, na, na, na});
        ambit::Tensor L2ab_sa =
            ambit::Tensor::build(ambit::CoreTensor, "L2ab_sa", {na, na, na, na});
        ambit::Tensor L2bb_sa =
            ambit::Tensor::build(ambit::CoreTensor, "L2bb_sa", {na, na, na, na});
        for (auto& casscf_ref : sa_cas_ref) {
            L1a_sa("u, v") += casscf_ref.g1a()("u, v");
            L1b_sa("u, v") += casscf_ref.g1b()("u, v");
            L2aa_sa("u, v, x, y") += casscf_ref.g2aa()("u, v, x, y");
            L2ab_sa("u, v, x, y") += casscf_ref.g2ab()("u, v, x, y");
            L2bb_sa("u, v, x, y") += casscf_ref.g2bb()("u, v, x, y");
        }
        // sa_cas.set_g2ab(L2ab);
        // sa_cas.set_g2bb(L2bb);

        /// You need to first get SA RDM and use those to convert to
        /// sa-cumulants.
        L1a_sa.scale(1.0 / nstates_);
        L1b_sa.scale(1.0 / nstates_);
        L2aa_sa.scale(1.0 / nstates_);
        L2ab_sa.scale(1.0 / nstates_);
        L2bb_sa.scale(1.0 / nstates_);

        L2aa_sa("pqrs") -= L1a_sa("pr") * L1a_sa("qs");
        L2aa_sa("pqrs") += L1a_sa("ps") * L1a_sa("qr");

        L2ab_sa("pqrs") -= L1a_sa("pr") * L1b_sa("qs");

        L2bb_sa("pqrs") -= L1b_sa("pr") * L1b_sa("qs");
        L2bb_sa("pqrs") += L1b_sa("ps") * L1b_sa("qr");

        sa_cas.set_L1a(L1a_sa);
        sa_cas.set_L1b(L1b_sa);
        sa_cas.set_L2aa(L2aa_sa);
        sa_cas.set_L2ab(L2ab_sa);
        sa_cas.set_L2bb(L2bb_sa);
    }
    sa_ref_ = sa_cas;

    return E_sa_casscf;
}
} // namespace forte
