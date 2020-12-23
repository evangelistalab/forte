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

#include <cmath>

#include "ambit/blocked_tensor.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/forte_options.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "semi_canonicalize.h"

using namespace psi;

namespace forte {

using namespace ambit;

SemiCanonical::SemiCanonical(std::shared_ptr<MOSpaceInfo> mo_space_info,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<ForteOptions> foptions, bool quiet_banner)
    : mo_space_info_(mo_space_info), ints_(ints) {

    if (!quiet_banner) {
        print_method_banner({"Semi-Canonical Orbitals",
                             "Chenyang Li, Jeffrey B. Schriber and Francesco A. Evangelista"});
    }

    // initialize the dimension objects
    read_options(foptions);
    startup();
}

void SemiCanonical::read_options(std::shared_ptr<ForteOptions> foptions) {
    inactive_mix_ = foptions->get_bool("SEMI_CANONICAL_MIX_INACTIVE");
    active_mix_ = foptions->get_bool("SEMI_CANONICAL_MIX_ACTIVE");

    // compute thresholds from options
    double econv = foptions->get_double("E_CONVERGENCE");
    threshold_tight_ = (econv < 1.0e-12) ? 1.0e-12 : econv;
    if (ints_->integral_type() == Cholesky) {
        double cd_tlr = foptions->get_double("CHOLESKY_TOLERANCE");
        threshold_tight_ = (threshold_tight_ < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_tight_;
    }
    threshold_loose_ = 10.0 * threshold_tight_;

    auto rconv = foptions->get_double("R_CONVERGENCE");
    auto dconv = foptions->get_double("D_CONVERGENCE");
    threshold_1rdm_ = rconv > dconv ? rconv : dconv;

    fix_orbital_success_ = true;
}

void SemiCanonical::startup() {
    nirrep_ = mo_space_info_->nirrep();
    nmopi_ = mo_space_info_->dimension("ALL");
    nact_ = mo_space_info_->size("ACTIVE");

    // Preapare orbital rotation matrix, which transforms all MOs
    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi_, nmopi_);
    Ub_ = std::make_shared<psi::Matrix>("Ub", nmopi_, nmopi_);

    // Preapare orbital rotation matrix, which transforms only active MOs
    Ua_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
    Ub_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

    // Initialize U to identity
    set_U_to_identity();

    // Get the list of elementary spaces
    auto composite_spaces = mo_space_info_->composite_space_names();
    auto docc_names = inactive_mix_ ? std::vector<std::string>{"INACTIVE_DOCC"}
                                    : composite_spaces["INACTIVE_DOCC"];
    auto actv_names = active_mix_ ? std::vector<std::string>{"ACTIVE"} : composite_spaces["ACTIVE"];
    auto uocc_names = inactive_mix_ ? std::vector<std::string>{"INACTIVE_UOCC"}
                                    : composite_spaces["INACTIVE_UOCC"];

    std::vector<std::string> space_names(docc_names);
    space_names.insert(space_names.end(), actv_names.begin(), actv_names.end());
    space_names.insert(space_names.end(), uocc_names.begin(), uocc_names.end());

    // Form dimension map
    for (std::string& space : space_names) {
        mo_dims_[space] = mo_space_info_->dimension(space);
    }

    // Compute the offset of GAS spaces within the ACTIVE
    for (const std::string& space : actv_names) {
        actv_offsets_[space] = psi::Dimension(nirrep_);
    }
    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        for (const std::string& space : actv_names) {
            actv_offsets_[space][h] = offset;
            offset += mo_dims_[space][h];
        }
    }
}

void SemiCanonical::set_U_to_identity() {
    Ua_->identity();
    Ub_->identity();

    Ua_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });

    Ub_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });
}

RDMs SemiCanonical::semicanonicalize(RDMs& rdms, const bool& build_fock, const bool& transform) {
    local_timer timer;

    // Build the Fock matrix from ForteIntegral
    if (build_fock) {
        ints_->make_fock_matrix(rdms.g1a(), rdms.g1b());
        outfile->Printf("\n  Took %8.6f s to build Fock matrix", timer.get());
    }

    // Prepare the Fock matrix
    // If doing natural orbitals, the active part of Fock will be 1-RDM
    prepare_fock(rdms);

    // Check Fock matrix
    if (check_fock_matrix()) {
        set_U_to_identity();
        outfile->Printf("\n  Orbitals are already semi-canonicalized.");
    } else {
        // Build transformation matrices from diagononalizing blocks in F
        build_transformation_matrices();

        // Retransform integrals and RDMs
        if (transform) {
            ints_->rotate_orbitals(Ua_, Ub_);
            rdms = rdms.rotate(Ua_t_, Ub_t_);
        }
        print_timing("semi-canonicalizing orbitals", timer.get());
    }

    return rdms;
}

void SemiCanonical::prepare_fock(RDMs& rdms) {
    auto fock_a = ints_->get_fock_a(false);
    auto fock_b = ints_->get_fock_b(false);

    if (!natural_orb_) {
        Fa_ = fock_a;
        Fb_ = fock_b;
    } else {
        auto relative_mos = mo_space_info_->relative_mo("ACTIVE");

        auto fill_data = [&](ambit::Tensor t, psi::SharedMatrix m) {
            t.citerate([&](const std::vector<size_t>& i, const double& value) {
                size_t h0, p0, h1, p1;
                std::tie(h0, p0) = relative_mos[i[0]];
                std::tie(h1, p1) = relative_mos[i[1]];
                if (h0 == h1) {
                    m->set(h0, p0, p1, value);
                }
            });
        };

        Fa_ = fock_a->clone();
        auto g1a = rdms.g1a();
        fill_data(g1a, Fa_); // fill Fa_ active with alpha 1-RDMs

        if (fock_a == fock_b and rdms.g1_spin_diff() < threshold_1rdm_) {
            Fb_ = Fa_;
        } else {
            Fb_ = fock_b->clone();
            auto g1b = rdms.g1b();
            fill_data(g1b, Fb_); // fill Fb_ active with beta 1-RDMs
        }
    }
}

bool SemiCanonical::check_fock_matrix() {
    print_h2("Checking Fock Matrix Diagonal Blocks");
    if (natural_orb_) {
        outfile->Printf("\n    Natural orbitals requested:");
        outfile->Printf("\n    Checking 1-RDM instead of Fock for active orbitals.\n");
    }

    std::vector<std::string> spin_cases{"alpha"};
    if (Fa_ != Fb_)
        spin_cases.push_back("beta");

    int width = 18 + 2 + 13 + 2 + 13;
    std::string dash(width, '-');
    outfile->Printf("\n    %s  %5c%s%5c  %4c%s", "Off-Diag. Elements", ' ', "Max", ' ', ' ',
                    "2-Norm");
    outfile->Printf("\n    %s", dash.c_str());

    for (const std::string& spin : spin_cases) {
        auto& fock = (spin == "alpha") ? Fa_ : Fb_;

        if (spin_cases.size() == 2) {
            outfile->Printf("\n    %s spin", spin.c_str());
            outfile->Printf("\n    %s", dash.c_str());
        }

        // loop over orbital spaces
        for (const auto& name_dim_pair : mo_dims_) {
            std::string name = name_dim_pair.first;
            psi::Dimension npi = name_dim_pair.second;
            if (npi.sum() == 0)
                continue;

            // grab Fock matrix of this diagonal block
            auto slice = mo_space_info_->range(name);
            auto Fsub = fock->get_block(slice, slice);
            Fsub->set_name("Fock " + name + " " + spin);

            Fsub->zero_diagonal();
            double Fmax = Fsub->absmax();
            double Fnorm = std::sqrt(Fsub->sum_of_squares());

            outfile->Printf("\n    %-18s  %13.10f  %13.10f", name.c_str(), Fmax, Fnorm);

            // check threshold
            double threshold_norm = npi.sum() * (npi.sum() - 1) * threshold_tight_;
            bool Fdo = Fmax > threshold_loose_ or Fnorm > threshold_norm;
            checked_results_[name + spin] = Fdo;
        }

        outfile->Printf("\n    %s", dash.c_str());
    }

    // return if orbitals are already semi-canonicalized
    return std::all_of(checked_results_.begin(), checked_results_.end(),
                       [](const auto& p) { return !p.second; });
}

void SemiCanonical::build_transformation_matrices() {
    // Diagonalize the diagonal blocks of the Fock matrix

    // set Ua and Ub to identity by default
    set_U_to_identity();

    std::vector<std::string> spin_cases{"alpha"};
    if (Fa_ != Fb_)
        spin_cases.push_back("beta");

    for (const std::string& spin : spin_cases) {
        bool is_alpha = (spin == "alpha");

        auto& fock = is_alpha ? Fa_ : Fb_;
        auto& U = is_alpha ? Ua_ : Ub_;

        // loop over orbital spaces
        for (const auto& name_dim_pair : mo_dims_) {
            const std::string& name = name_dim_pair.first;
            psi::Dimension npi = name_dim_pair.second;
            if (npi.sum() == 0)
                continue;

            if (checked_results_[name + spin]) {
                // build Fock matrix of this diagonal block
                auto slice = mo_space_info_->range(name);
                auto Fsub = fock->get_block(slice, slice);
                Fsub->set_name("Fock " + name + " " + spin);

                // diagonalize this Fock block
                auto Usub = std::make_shared<psi::Matrix>("U " + name + " " + spin, npi, npi);
                auto evals = std::make_shared<psi::Vector>("evals" + name + " " + spin, npi);
                if (natural_orb_ and mo_space_info_->contained_in_space(name, "ACTIVE")) {
                    Fsub->diagonalize(Usub, evals, descending);
                } else {
                    Fsub->diagonalize(Usub, evals, ascending);
                }

                // fill in Ua or Ub
                U->set_block(slice, slice, Usub);
            }
        }

        // keep phase and order unchanged
        fix_orbital_success_ &= ints_->fix_orbital_phases(U, is_alpha);

        // fill in UData
        auto& Ut = is_alpha ? Ua_t_ : Ub_t_;
        fill_Uactv(U, Ut);
    }

    if (Fa_ == Fb_) {
        Ub_->copy(Ua_);
        Ub_t_.copy(Ua_t_);
    }
}

void SemiCanonical::fill_Uactv(psi::SharedMatrix U, ambit::Tensor Ut) {
    auto actv_names = active_mix_ ? std::vector<std::string>{"ACTIVE"}
                                  : mo_space_info_->composite_space_names()["ACTIVE"];
    auto& Ut_data = Ut.data();

    for (const std::string& name : actv_names) {
        auto size = mo_space_info_->size(name);
        if (size == 0)
            continue;

        auto pos = mo_space_info_->pos_in_space(name, "ACTIVE");
        auto relative_mos = mo_space_info_->relative_mo(name);

        for (size_t p = 0; p < size; ++p) {
            size_t hp, np;
            std::tie(hp, np) = relative_mos[p];

            for (size_t q = 0; q < size; ++q) {
                size_t hq, nq;
                std::tie(hq, nq) = relative_mos[q];
                if (hp != hq)
                    continue;

                Ut_data[pos[p] * nact_ + pos[q]] = U->get(hp, np, nq);
            }
        }
    }
}
} // namespace forte
