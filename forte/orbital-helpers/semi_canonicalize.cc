/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "base_classes/scf_info.h"

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
                             std::shared_ptr<SCFInfo> scf_info, bool inactive_mix, bool active_mix,
                             double threshold, bool quiet)
    : mo_space_info_(mo_space_info), ints_(ints), scf_info_(scf_info), print_(not quiet),
      inactive_mix_(inactive_mix), active_mix_(active_mix), threshold_tight_(threshold),
      fix_orbital_success_(true) {
    startup(); // initialize the dimension objects
}

void SemiCanonical::startup() {
    nirrep_ = mo_space_info_->nirrep();
    nmopi_ = mo_space_info_->dimension("ALL");
    nact_ = mo_space_info_->size("ACTIVE");

    // Prepare orbital rotation matrix, which transforms all MOs
    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi_, nmopi_);
    Ub_ = std::make_shared<psi::Matrix>("Ub", nmopi_, nmopi_);

    // Prepare orbital rotation matrix, which transforms only active MOs
    Ua_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
    Ub_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

    // Initialize U to identity
    set_U_to_identity();

    // Find the elementary blocks
    auto composite_spaces = mo_space_info_->composite_spaces_def();
    auto docc_names = inactive_mix_ ? std::vector<std::string>{"INACTIVE_DOCC"}
                                    : composite_spaces["INACTIVE_DOCC"];
    auto actv_names = active_mix_ ? std::vector<std::string>{"ACTIVE"} : composite_spaces["ACTIVE"];
    auto uocc_names = inactive_mix_ ? std::vector<std::string>{"INACTIVE_UOCC"}
                                    : composite_spaces["INACTIVE_UOCC"];

    std::vector<std::string> space_names(docc_names);
    space_names.insert(space_names.end(), actv_names.begin(), actv_names.end());
    space_names.insert(space_names.end(), uocc_names.begin(), uocc_names.end());

    // Form dimension map
    for (const std::string& space : space_names) {
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

    threshold_loose_ = 10.0 * threshold_tight_;
}

void SemiCanonical::set_U_to_identity() {
    Ua_->identity();
    Ub_->identity();

    Ua_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });

    Ub_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });
}

void SemiCanonical::semicanonicalize(std::shared_ptr<RDMs> rdms, bool build_fock,
                                     ActiveOrbitalType orb_type, bool transform) {
    timer t_semi("semicanonicalize orbitals");

    print_h2("Semicanonicalize Orbitals");
    auto true_or_false = [](bool x) { return x ? "TRUE" : "FALSE"; };
    outfile->Printf("\n    MIX INACTIVE ORBITALS         %s", true_or_false(inactive_mix_));
    outfile->Printf("\n    MIX GAS ACTIVE ORBITALS       %s", true_or_false(active_mix_));

    // build Fock matrix
    if (build_fock) {
        timer t_fock("build Fock");
        // use spin-free 1-RDM
        // TODO: this need to be changed if UHF reference is available
        auto g1 = rdms->SF_G1().clone();
        g1.scale(0.5);
        ints_->make_fock_matrix(g1, g1);
    }

    // build transformation matrix based on Fock or 1-RDM
    bool already_semi = check_orbitals(rdms, orb_type);
    build_transformation_matrices(already_semi);
    if (transform and (not already_semi)) {
        scf_info_->rotate_orbitals(Ua_, Ub_);
        // ints_->rotate_orbitals(Ua_, Ub_);
        rdms->rotate(Ua_t_, Ub_t_);
    }
    if (print_)
        print_timing("orbital canonicalization", t_semi.stop());
}

bool SemiCanonical::check_orbitals(std::shared_ptr<RDMs> rdms, ActiveOrbitalType orb_type) {
    bool semi = true;

    // print orbitals requested
    for (const auto& [name, npi] : mo_dims_) {
        if (mo_space_info_->contained_in_space(name, "ACTIVE")) {
            if (npi.sum() != 0) {
                outfile->Printf("\n    %-15s%15c%s", name.c_str(), ' ',
                                orb_type.toString().c_str());
            }
        } else {
            outfile->Printf("\n    %-15s%15c%s", name.c_str(), ' ', "CANONICAL");
        }
    }

    // prepare data blocks
    prepare_matrix_blocks(rdms, orb_type);

    // check data blocks
    int width = 18 + 2 + 13 + 2 + 13;
    std::string dash(width, '-');
    if (print_) {
        outfile->Printf("\n\n    %s  %5c%s%5c  %4c%s", "Off-Diag. Elements", ' ', "Max", ' ', ' ',
                        "2-Norm");
        outfile->Printf("\n    %s", dash.c_str());
    }

    for (const auto& [name, M_orig] : mats_) {
        auto M = M_orig->clone();
        M->zero_diagonal();
        double v_max = M->absmax();
        double v_norm = std::sqrt(M->sum_of_squares());

        if (print_)
            outfile->Printf("\n    %-18s  %13.10f  %13.10f", name.c_str(), v_max, v_norm);

        // check threshold
        int nrow = M->nrow();
        double threshold_norm = nrow * (nrow - 1) * threshold_tight_;
        checked_results_[name] = v_max > threshold_loose_ or v_norm > threshold_norm;
        if (checked_results_[name]) {
            semi = false;
        }
    }

    if (print_) {
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n\n    Canonicalization test %s\n", semi ? "passed" : "failed");
    }

    return semi;
}

void SemiCanonical::prepare_matrix_blocks(std::shared_ptr<RDMs> rdms, ActiveOrbitalType orb_type) {
    mats_.clear();

    // Fock alpha should be the same as beta
    // TODO: this need to be changed if UHF reference is available
    auto fock = ints_->get_fock_a(false);
    if (fock != ints_->get_fock_b(false)) {
        throw std::runtime_error("Currently impossible to semicanonicalize unrestricted orbitals!");
    }

    // 1-RDM in Matrix format
    auto d1 = rdms->SF_G1mat(mo_space_info_->dimension("ACTIVE"));
    auto docc_offset = mo_space_info_->dimension("INACTIVE_DOCC");

    // loop over orbital spaces
    for (const auto& [name, npi] : mo_dims_) {
        // filter out zero dimension blocks
        if (npi.sum() == 0)
            continue;

        // fill data
        auto slice = mo_space_info_->range(name);
        if (mo_space_info_->contained_in_space(name, "ACTIVE")) {
            if (orb_type == ActiveOrbitalType::unspecified) {
                continue;
            } else if (orb_type == ActiveOrbitalType::natural) {
                // For natural orbitals, diagonalize the 1-RDM in the active space
                auto actv_slice =
                    psi::Slice(slice.begin() - docc_offset, slice.end() - docc_offset);
                mats_[name] = d1->get_block(actv_slice, actv_slice);
                mats_[name]->set_name("D1 " + name);
            } else {
                // By default, diagonalize the Fock in the active space
                mats_[name] = fock->get_block(slice, slice);
                mats_[name]->set_name("Fock " + name);
            }
        } else {
            // for all other spaces always diagonalize the Fock matrix
            mats_[name] = fock->get_block(slice, slice);
            mats_[name]->set_name("Fock " + name);
        }
    }
}

void SemiCanonical::build_transformation_matrices(const bool& semi) {
    // set Ua and Ub to identity by default
    set_U_to_identity();

    if (semi) {
        outfile->Printf("\n  Orbitals are already semicanonicalized.");
        return;
    }

    // loop over data blocks
    for (const auto& [name, M] : mats_) {
        if (checked_results_[name]) {
            // natural orbital in descending order, canonical orbital in ascending order
            bool ascending = M->name().find("Fock") != std::string::npos;

            // diagonalize this block
            auto Usub = std::make_shared<psi::Matrix>("U " + name, M->rowspi(), M->colspi());
            auto evals = std::make_shared<psi::Vector>("evals" + name, M->rowspi());
            M->diagonalize(Usub, evals, ascending ? psi::ascending : psi::descending);

            // fill in Ua or Ub
            auto slice = mo_space_info_->range(name);
            Ua_->set_block(slice, slice, Usub);
        }
    }

    // keep phase and order unchanged
    if (!inactive_mix_)
        fix_orbital_success_ = ints_->fix_orbital_phases(Ua_, true);

    // fill in Ua_t_
    fill_Uactv(Ua_, Ua_t_);

    // pass to Ub
    Ub_->copy(Ua_);
    Ub_t_.copy(Ua_t_);
}

void SemiCanonical::fill_Uactv(const std::shared_ptr<psi::Matrix>& U, ambit::Tensor& Ut) {
    auto actv_names = active_mix_ ? std::vector<std::string>{"ACTIVE"}
                                  : mo_space_info_->composite_spaces_def().at("ACTIVE");
    auto& Ut_data = Ut.data();

    for (const std::string& name : actv_names) {
        auto size = mo_space_info_->size(name);
        if (size == 0)
            continue;

        auto pos = mo_space_info_->pos_in_space(name, "ACTIVE");
        auto relative_mos = mo_space_info_->relative_mo(name);
        for (size_t p = 0; p < size; ++p) {
            const auto& [hp, np] = relative_mos[p];
            for (size_t q = 0; q < size; ++q) {
                const auto& [hq, nq] = relative_mos[q];
                if (hp != hq)
                    continue;
                Ut_data[pos[p] * nact_ + pos[q]] = U->get(hp, np, nq);
            }
        }
    }
}
} // namespace forte
