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

    // 0. initialize the dimension objects
    startup();

    // compute thresholds from options
    double econv = foptions->get_double("E_CONVERGENCE");
    threshold_tight_ = (econv < 1.0e-12) ? 1.0e-12 : econv;
    if (ints_->integral_type() == Cholesky) {
        double cd_tlr = foptions->get_double("CHOLESKY_TOLERANCE");
        threshold_tight_ = (threshold_tight_ < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_tight_;
    }
    threshold_loose_ = 10.0 * threshold_tight_;
}

void SemiCanonical::startup() {
    // some basics
    nirrep_ = mo_space_info_->nirrep();
    nmopi_ = mo_space_info_->dimension("ALL");
    ncmo_ = mo_space_info_->size("CORRELATED");
    nact_ = mo_space_info_->size("ACTIVE");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");
    fdocc_ = mo_space_info_->dimension("FROZEN_DOCC");
    rdocc_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->dimension("ACTIVE");

    // Preapare orbital rotation matrix, which transforms all MOs
    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi_, nmopi_);
    Ub_ = std::make_shared<psi::Matrix>("Ub", nmopi_, nmopi_);

    // Preapare orbital rotation matrix, which transforms only active MOs
    Ua_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
    Ub_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

    // Initialize U to identity
    set_U_to_identity();

    // Get the list of elementary spaces
    std::vector<std::string> space_names = mo_space_info_->space_names();

    // dimension map
    for (std::string& space : space_names) {
        mo_dims_[space] = mo_space_info_->dimension(space);
    }

    // index map and offsets map
    auto offset_dim = psi::Dimension(nirrep_);
    for (std::string& space : space_names) {
        auto dim = mo_dims_[space];
        mo_idx_[space] = idx_space(dim, offset_dim);
        offsets_[space] = offset_dim;
        offset_dim += dim;
    }

    // Compute the offset within the GAS spaces
    for (std::string& space : space_names) {
        if (space.find("GAS") != std::string::npos) {
            actv_offsets_[space] = std::vector<int>(nirrep_, 0);
        }
    }
    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        for (std::string& space : space_names) {
            if (space.find("GAS") != std::string::npos) {
                actv_offsets_[space][h] = offset;
                offset += mo_dims_[space][h];
            }
        }
    }
}

std::vector<std::vector<size_t>> SemiCanonical::idx_space(const psi::Dimension& npi,
                                                          const psi::Dimension& bpi) {
    std::vector<std::vector<size_t>> out(nirrep_, std::vector<size_t>());

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        offset += bpi[h];
        for (int i = 0; i < npi[h]; ++i) {
            out[h].emplace_back(offset + i);
        }
        offset += nmopi_[h] - bpi[h];
    }

    return out;
}

RDMs SemiCanonical::semicanonicalize(RDMs& rdms, const int& max_rdm_level, const bool& build_fock,
                                     const bool& transform) {
    local_timer SemiCanonicalize;

    // 1. Build the Fock matrix from ForteIntegral
    if (build_fock) {
        local_timer FockTime;
        ints_->make_fock_matrix(rdms.g1a(), rdms.g1b());
        outfile->Printf("\n  Took %8.6f s to build Fock matrix", FockTime.get());
    }

    // Check Fock matrix
    bool semi = check_fock_matrix();

    if (semi) {
        outfile->Printf("\n  Orbitals are already semicanonicalized.");
        set_U_to_identity();
    } else {
        // 2. Build transformation matrices from diagononalizing blocks in F
        build_transformation_matrices();

        // 3. Retransform integrals and cumulants/RDMs
        if (transform) {
            ints_->rotate_orbitals(Ua_, Ua_);
            rdms = transform_rdms(Ua_t_, Ua_t_, rdms, max_rdm_level);
        }
        print_timing("semi-canonicalization", SemiCanonicalize.get());
    }

    return rdms;
}

bool SemiCanonical::check_fock_matrix() {
    print_h2("Checking Fock Matrix Diagonal Blocks");
    bool semi = true;

    auto fock_a = ints_->get_fock_a(false);
    auto fock_b = ints_->get_fock_b(false);
    std::vector<std::string> spin_cases{"alpha"};
    if (fock_a != fock_b)
        spin_cases.push_back("beta");

    int width = 18 + 2 + 13 + 2 + 13;
    std::string dash(width, '-');
    outfile->Printf("\n    %s  %5c%s%5c  %4c%s", "Off-Diag. Elements", ' ', "Max", ' ', ' ',
                    "2-Norm");
    outfile->Printf("\n    %s", dash.c_str());

    for (const std::string& spin : spin_cases) {
        auto& fock = (spin == "alpha") ? fock_a : fock_b;

        // loop over orbital spaces
        for (const auto& name_dim_pair : mo_dims_) {
            std::string name = name_dim_pair.first;
            psi::Dimension npi = name_dim_pair.second;

            // grab Fock matrix of this diagonal block
            psi::Slice slice(offsets_[name], offsets_[name] + npi);
            auto Fsub = fock->get_block(slice, slice);
            Fsub->set_name("Fock " + name + " " + spin);

            Fsub->zero_diagonal();
            double Fmax = Fsub->absmax();
            double Fnorm = std::sqrt(Fsub->sum_of_squares());

            outfile->Printf("\n    %-18s  %13.10f  %13.10f", name.c_str(), Fmax, Fnorm);

            // check threshold
            double threshold_norm = npi.sum() * (npi.sum() - 1) * threshold_tight_;
            bool Fdo = (Fmax <= threshold_loose_ && Fnorm <= threshold_norm) ? false : true;
            checked_results_[name + spin] = Fdo;
            if (Fdo) {
                semi = false;
            }
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    return semi;
}

void SemiCanonical::set_U_to_identity() {
    Ua_->identity();
    Ub_->identity();

    Ua_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });

    Ub_t_.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });
}

void SemiCanonical::build_transformation_matrices() {
    // 2. Diagonalize the diagonal blocks of the Fock matrix

    // set Ua and Ub to identity by default
    set_U_to_identity();

    auto fock_a = ints_->get_fock_a(false);
    auto fock_b = ints_->get_fock_b(false);

    std::vector<std::string> spin_cases{"alpha"};
    if (fock_a != fock_b)
        spin_cases.push_back("beta");

    for (const std::string& spin : spin_cases) {
        auto& fock = (spin == "alpha") ? fock_a : fock_b;
        auto& U = (spin == "alpha") ? Ua_ : Ub_;

        // loop over orbital spaces
        for (const auto& name_dim_pair : mo_dims_) {
            const std::string& name = name_dim_pair.first;
            psi::Dimension npi = name_dim_pair.second;

            if (checked_results_[name + spin]) {
                // build Fock matrix of this diagonal block
                psi::Slice slice(offsets_[name], offsets_[name] + npi);
                auto Fsub = fock->get_block(slice, slice);
                Fsub->set_name("Fock " + name + " " + spin);

                // diagonalize this Fock block
                auto Usub = std::make_shared<psi::Matrix>("U " + name + " " + spin, npi, npi);
                auto evals = std::make_shared<psi::Vector>("evals" + name + " " + spin, npi);
                Fsub->diagonalize(Usub, evals);

                // fill in Ua or Ub
                for (size_t h = 0; h < nirrep_; ++h) {
                    int offset = offsets_[name][h];
                    for (int i = 0; i < npi[h]; ++i) {
                        for (int j = 0; j < npi[h]; ++j) {
                            U->set(h, offset + i, offset + j, Usub->get(h, i, j));
                        }
                    }
                }
            }
        }
    }

    // keep phase and order unchanged
    ints_->fix_orbital_phases(Ua_, true);
    ints_->fix_orbital_phases(Ub_, false);

    // fill in UaData and UbData
    for (const std::string& spin : spin_cases) {
        auto& U = (spin == "alpha") ? Ua_ : Ub_;
        auto& UData = (spin == "alpha") ? Ua_t_.data() : Ub_t_.data();

        for (const auto& name_dim_pair : mo_dims_) {
            const std::string& name = name_dim_pair.first;
            psi::Dimension npi = name_dim_pair.second;

            if (name.find("GAS") == std::string::npos)
                continue;

            for (size_t h = 0; h < nirrep_; ++h) {
                int actv_off = actv_offsets_[name][h];
                int offset = offsets_[name][h];

                for (int u = 0; u < npi[h]; ++u) {
                    int nu = actv_off + u;
                    int mu = offset + u;

                    for (int v = 0; v < npi[h]; ++v) {
                        UData[nu * nact_ + actv_off + v] = U->get(h, mu, offset + v);
                    }
                }
            }
        }
    }

    if (fock_a == fock_b) {
        Ub_->copy(Ua_);
        Ub_t_.copy(Ua_t_);
    }
}

RDMs SemiCanonical::transform_rdms(ambit::Tensor& Ua, ambit::Tensor& Ub, RDMs& rdms,
                                   const int& max_rdm_level) {
    if (max_rdm_level < 1)
        return RDMs();

    print_h2("RDMs Transformation to Semicanonical Basis");

    if (rdms.ms_avg()) {
        auto g1 = rdms.g1a();
        auto g1T = ambit::Tensor::build(ambit::CoreTensor, "g1aT", {nact_, nact_});
        g1T("pq") = Ua("ap") * g1("ab") * Ua("bq");
        outfile->Printf("\n    Transformed 1 RDM.");
        if (max_rdm_level == 1) {
            return RDMs(true, g1T);
        }

        auto g2 = rdms.g2ab();
        auto g2T = ambit::Tensor::build(ambit::CoreTensor, "g2abT", {nact_, nact_, nact_, nact_});
        g2T("pQrS") = Ua("ap") * Ua("BQ") * g2("aBcD") * Ua("cr") * Ua("DS");
        outfile->Printf("\n    Transformed 2 RDM.");
        if (max_rdm_level == 2)
            return RDMs(true, g1T, g2T);

        auto g3 = rdms.g3aab();
        auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3aabT", std::vector<size_t>(6, nact_));
        g3T("pqRstU") =
            Ua("ap") * Ua("bq") * Ua("CR") * g3("abCijK") * Ua("is") * Ua("jt") * Ua("KU");
        outfile->Printf("\n    Transformed 3 RDM.");
        return RDMs(true, g1T, g2T, g3T);
    }

    // Transform the 1-cumulants
    ambit::Tensor g1a0 = rdms.g1a();
    ambit::Tensor g1b0 = rdms.g1b();

    ambit::Tensor g1aT = ambit::Tensor::build(ambit::CoreTensor, "g1aT", {nact_, nact_});
    ambit::Tensor g1bT = ambit::Tensor::build(ambit::CoreTensor, "g1bT", {nact_, nact_});

    g1aT("pq") = Ua("ap") * g1a0("ab") * Ua("bq");
    g1bT("PQ") = Ub("AP") * g1b0("AB") * Ub("BQ");

    outfile->Printf("\n    Transformed 1 RDMs.");

    if (max_rdm_level == 1)
        return RDMs(g1aT, g1bT);

    // the original 2-rdms
    ambit::Tensor g2aa0 = rdms.g2aa();
    ambit::Tensor g2ab0 = rdms.g2ab();
    ambit::Tensor g2bb0 = rdms.g2bb();

    //   aa spin
    auto g2Taa = ambit::Tensor::build(ambit::CoreTensor, "g2aaT", {nact_, nact_, nact_, nact_});
    g2Taa("pqrs") = Ua("ap") * Ua("bq") * g2aa0("abcd") * Ua("cr") * Ua("ds");

    //   ab spin
    auto g2Tab = ambit::Tensor::build(ambit::CoreTensor, "g2abT", {nact_, nact_, nact_, nact_});
    g2Tab("pQrS") = Ua("ap") * Ub("BQ") * g2ab0("aBcD") * Ua("cr") * Ub("DS");

    //   bb spin
    auto g2Tbb = ambit::Tensor::build(ambit::CoreTensor, "g2bbT", {nact_, nact_, nact_, nact_});
    g2Tbb("PQRS") = Ub("AP") * Ub("BQ") * g2bb0("ABCD") * Ub("CR") * Ub("DS");

    outfile->Printf("\n    Transformed 2 RDMs.");

    if (max_rdm_level == 2)
        return RDMs(g1aT, g1bT, g2Taa, g2Tab, g2Tbb);

    // Transform 3 cumulants
    ambit::Tensor g3aaa0 = rdms.g3aaa();
    ambit::Tensor g3aab0 = rdms.g3aab();
    ambit::Tensor g3abb0 = rdms.g3abb();
    ambit::Tensor g3bbb0 = rdms.g3bbb();

    auto g3Taaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaaT", std::vector<size_t>(6, nact_));
    g3Taaa("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * g3aaa0("abcijk") * Ua("is") * Ua("jt") * Ua("ku");

    auto g3Taab = ambit::Tensor::build(ambit::CoreTensor, "g3aabT", std::vector<size_t>(6, nact_));
    g3Taab("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * g3aab0("abCijK") * Ua("is") * Ua("jt") * Ub("KU");

    auto g3Tabb = ambit::Tensor::build(ambit::CoreTensor, "g3abbT", std::vector<size_t>(6, nact_));
    g3Tabb("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * g3abb0("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");

    auto g3Tbbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbbT", std::vector<size_t>(6, nact_));
    g3Tbbb("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * g3bbb0("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");

    outfile->Printf("\n    Transformed 3 RDMs.");

    return RDMs(g1aT, g1bT, g2Taa, g2Tab, g2Tbb, g3Taaa, g3Taab, g3Tabb, g3Tbbb);
}
} // namespace forte
