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
    ncmo_ = mo_space_info_->size("CORRELATED");
    nact_ = mo_space_info_->size("ACTIVE");
    nmopi_ = mo_space_info_->dimension("ALL");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");
    fdocc_ = mo_space_info_->dimension("FROZEN_DOCC");
    rdocc_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->dimension("ACTIVE");
    ruocc_ = mo_space_info_->dimension("RESTRICTED_UOCC");

    // Preapare orbital rotation matrix, which transforms all MOs
    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi_, nmopi_);
    Ub_ = std::make_shared<psi::Matrix>("Ub", nmopi_, nmopi_);

    // Preapare orbital rotation matrix, which transforms only active MOs
    Ua_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
    Ub_t_ = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

    // Initialize U to identity
    set_U_to_identity();

    // dimension map
    mo_dims_["core"] = rdocc_;
    mo_dims_["actv"] = actv_;
    mo_dims_["virt"] = ruocc_;

    // index map
    cmo_idx_["core"] = idx_space(rdocc_, psi::Dimension(std::vector<int>(nirrep_, 0)), ncmopi_);
    cmo_idx_["actv"] = idx_space(actv_, rdocc_, ncmopi_);
    cmo_idx_["virt"] = idx_space(ruocc_, rdocc_ + actv_, ncmopi_);

    // offsets map
    offsets_["core"] = fdocc_;
    offsets_["virt"] = fdocc_ + rdocc_ + actv_;
    offsets_["actv"] = fdocc_ + rdocc_;

    std::vector<int> actv_off;
    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        actv_off.emplace_back(offset);
        offset += actv_[h];
    }
    actv_offsets_["actv"] = actv_off;
}

std::vector<std::vector<size_t>> SemiCanonical::idx_space(const psi::Dimension& npi,
                                                          const psi::Dimension& bpi,
                                                          const psi::Dimension& tpi) {
    std::vector<std::vector<size_t>> out(nirrep_, std::vector<size_t>());

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        offset += bpi[h];
        for (int i = 0; i < npi[h]; ++i) {
            out[h].emplace_back(offset + i);
        }
        offset += tpi[h] - bpi[h];
    }

    return out;
}

void SemiCanonical::set_actv_dims(const psi::Dimension& actv_docc,
                                  const psi::Dimension& actv_virt) {
    // test actv_docc and actv_virt
    psi::Dimension actv = actv_docc + actv_virt;
    if (actv != actv_) {
        throw psi::PSIEXCEPTION("ACTIVE_DOCC and ACTIVE_VIRT do not add up to ACTIVE!");
    }

    // delete original active maps
    mo_dims_.erase("actv");
    cmo_idx_.erase("actv");
    offsets_.erase("actv");
    actv_offsets_.erase("actv");

    // save to class variables
    actv_docc_ = actv_docc;
    actv_virt_ = actv_virt;

    // active dimension map
    mo_dims_["actv_docc"] = actv_docc_;
    mo_dims_["actv_virt"] = actv_virt_;

    // active index map
    cmo_idx_["actv_docc"] = idx_space(actv_docc_, rdocc_, ncmopi_);
    cmo_idx_["actv_virt"] = idx_space(actv_virt_, rdocc_ + actv_docc, ncmopi_);

    // active offsets map
    offsets_["actv_docc"] = fdocc_ + rdocc_;
    offsets_["actv_virt"] = fdocc_ + rdocc_ + actv_docc_;

    std::vector<int> actvh_off, actvp_off;
    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        actvh_off.emplace_back(offset);
        offset += actv_docc[h];
        actvp_off.emplace_back(offset);
        offset += actv_virt[h];
    }
    actv_offsets_["actv_docc"] = actvh_off;
    actv_offsets_["actv_virt"] = actvp_off;
}

RDMs SemiCanonical::semicanonicalize(RDMs& rdms, const int& max_rdm_level, const bool& build_fock,
                                     const bool& transform) {
    local_timer SemiCanonicalize;

    // 1. Build the Fock matrix from ForteIntegral
    if (build_fock) {
        build_fock_matrix(rdms);
    }

    // Check Fock matrix
    bool semi = check_fock_matrix();

    if (semi) {
        outfile->Printf("\n  Orbitals are already semicanonicalized.");
        set_U_to_identity();
    } else {
        // 2. Build transformation matrices from diagononalizing blocks in F
        build_transformation_matrices(Ua_, Ub_, Ua_t_, Ub_t_);

        // 3. Retransform integrals and cumulants/RDMs
        if (transform) {
            ints_->rotate_orbitals(Ua_, Ua_);
            rdms = transform_rdms(Ua_t_, Ua_t_, rdms, max_rdm_level);
        }
        outfile->Printf("\n  SemiCanonicalize takes %8.6f s.", SemiCanonicalize.get());
    }
    return rdms;
}

void SemiCanonical::build_fock_matrix(RDMs& rdms) {
    // 1. Build the Fock matrix

    psi::SharedMatrix Da(new psi::Matrix("Da", ncmo_, ncmo_));
    psi::SharedMatrix Db(new psi::Matrix("Db", ncmo_, ncmo_));

    auto L1a = tensor_to_matrix(rdms.g1a(), actv_);
    auto L1b = tensor_to_matrix(rdms.g1b(), actv_);

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        // core block (diagonal)
        for (int i = 0; i < rdocc_[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Da->set(offset + u, offset + v, L1a->get(h, u, v));
                Db->set(offset + u, offset + v, L1b->get(h, u, v));
            }
        }

        offset += ncmopi_[h] - rdocc_[h];
    }

    local_timer FockTime;
    ints_->make_fock_matrix(Da, Db);
    outfile->Printf("\n  Took %8.6f s to build Fock matrix", FockTime.get());
}

bool SemiCanonical::check_fock_matrix() {
    print_h2("Checking Fock Matrix Diagonal Blocks");
    bool semi = true;

    int width = 18 + 2 + 13 + 2 + 13;
    std::string dash(width, '-');
    outfile->Printf("\n    %s  %5c%s%5c  %4c%s", "Off-Diag. Elements", ' ', "Max", ' ', ' ',
                    "2-Norm");
    outfile->Printf("\n    %s", dash.c_str());

    // loop over orbital spaces
    for (const auto& name_dim_pair : mo_dims_) {
        std::string name = name_dim_pair.first;
        std::string name_a = "Fa " + name;
        std::string name_b = "Fb " + name;
        psi::Dimension npi = name_dim_pair.second;

        // build Fock matrix of this diagonal block
        psi::SharedMatrix Fa(new psi::Matrix(name_a, npi, npi));
        psi::SharedMatrix Fb(new psi::Matrix(name_b, npi, npi));

        for (size_t h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < npi[h]; ++i) {
                for (int j = 0; j < npi[h]; ++j) {
                    Fa->set(h, i, j, ints_->get_fock_a(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                    Fb->set(h, i, j, ints_->get_fock_b(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                }
            }
        }

        // zero diagonal elements
        Fa->zero_diagonal();
        Fb->zero_diagonal();

        // max value
        double Famax = Fa->absmax();
        double Fbmax = Fb->absmax();

        // 2-norm
        double Fanorm = std::sqrt(Fa->sum_of_squares());
        double Fbnorm = std::sqrt(Fb->sum_of_squares());

        // printing
        outfile->Printf("\n    %-18s  %13.10f  %13.10f", name_a.c_str(), Famax, Fanorm);
        outfile->Printf("\n    %-18s  %13.10f  %13.10f", name_b.c_str(), Fbmax, Fbnorm);
        outfile->Printf("\n    %s", dash.c_str());

        // check threshold
        double threshold_norm = npi.sum() * (npi.sum() - 1) * threshold_tight_;
        bool FaDo = (Famax <= threshold_loose_ && Fanorm <= threshold_norm) ? false : true;
        bool FbDo = (Fbmax <= threshold_loose_ && Fbnorm <= threshold_norm) ? false : true;
        bool FDo = FaDo && FbDo;
        checked_results_[name] = FDo;
        if (FDo) {
            semi = false;
        }
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

void SemiCanonical::build_transformation_matrices(psi::SharedMatrix& Ua, psi::SharedMatrix& Ub,
                                                  ambit::Tensor& Ua_t, ambit::Tensor& Ub_t) {
    // 2. Diagonalize the diagonal blocks of the Fock matrix

    // set Ua and Ub to identity by default
    Ua->identity();
    Ub->identity();
    std::vector<double> UaData(nact_ * nact_, 0.0);
    std::vector<double> UbData(nact_ * nact_, 0.0);
    for (size_t i = 0; i < nact_; ++i) {
        UaData[i * nact_ + i] = 1.0;
        UbData[i * nact_ + i] = 1.0;
    }

    // loop over orbital spaces
    for (const auto& name_dim_pair : mo_dims_) {
        std::string name = name_dim_pair.first;
        std::string name_a = "Fock " + name + " alpha";
        std::string name_b = "Fock " + name + " beta";
        psi::Dimension npi = name_dim_pair.second;
        bool FockDo = checked_results_[name];

        if (FockDo) {
            // build Fock matrix of this diagonal block
            psi::SharedMatrix Fa(new psi::Matrix(name_a, npi, npi));
            psi::SharedMatrix Fb(new psi::Matrix(name_b, npi, npi));

            for (size_t h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < npi[h]; ++i) {
                    for (int j = 0; j < npi[h]; ++j) {
                        Fa->set(h, i, j,
                                ints_->get_fock_a(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                        Fb->set(h, i, j,
                                ints_->get_fock_b(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                    }
                }
            }

            // diagonalize this Fock block
            psi::SharedMatrix UsubA(new psi::Matrix("Ua " + name, npi, npi));
            psi::SharedMatrix UsubB(new psi::Matrix("Ub " + name, npi, npi));
            psi::SharedVector evalsA(new Vector("evals a " + name, npi));
            psi::SharedVector evalsB(new Vector("evals b " + name, npi));
            Fa->diagonalize(UsubA, evalsA);
            Fb->diagonalize(UsubB, evalsB);

            // fill in Ua and Ub
            for (size_t h = 0; h < nirrep_; ++h) {
                int offset = offsets_[name][h];
                for (int i = 0; i < npi[h]; ++i) {
                    for (int j = 0; j < npi[h]; ++j) {
                        Ua->set(h, offset + i, offset + j, UsubA->get(h, i, j));
                        Ub->set(h, offset + i, offset + j, UsubB->get(h, i, j));
                    }
                }
            }
        }
    }

    // keep phase and order unchanged
    ints_->fix_orbital_phases(Ua, true);
    ints_->fix_orbital_phases(Ub, false);

    // fill in UaData and UbData
    for (const auto& name_dim_pair : mo_dims_) {
        std::string name = name_dim_pair.first;
        psi::Dimension npi = name_dim_pair.second;

        if (checked_results_[name] and (name.find("actv") != std::string::npos)) {
            for (size_t h = 0; h < nirrep_; ++h) {
                int actv_off = actv_offsets_[name][h];
                int offset = offsets_[name][h];
                for (int u = 0; u < npi[h]; ++u) {
                    for (int v = 0; v < npi[h]; ++v) {
                        int nu = actv_off + u;
                        int nv = actv_off + v;
                        UaData[nu * nact_ + nv] = Ua->get(h, offset + u, offset + v);
                        UbData[nu * nact_ + nv] = Ub->get(h, offset + u, offset + v);
                    }
                }
            }
        }
    }

    // copy active data to ambit tensors
    // temporary fix of DF integrals until both Ca and Cb are considered in DF integrals.
    auto type = ints_->integral_type();
    if (type == DF || type == DiskDF || type == Cholesky) {
        Ub->copy(Ua);
        Ua_t.data() = UaData;
        Ub_t.data() = UaData;
    } else {
        Ua_t.data() = UaData;
        Ub_t.data() = UbData;
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

void SemiCanonical::fix_orbital_phase(psi::SharedMatrix& Ua, const psi::SharedMatrix& Ca) {
    // build MO overlap matrix (old by new)
    auto Cnew = psi::linalg::doublet(Ca, Ua, false, false);
    Cnew->set_name("MO coefficients (new)");

    auto Smo = psi::linalg::triplet(Ca, ints_->wfn()->S(), Cnew, true, false, false);
    Smo->set_name("MO overlap (old by new)");

    // transformation matrix
    auto T = Ua->clone();
    T->set_name("Reordering matrix");
    T->zero();

    for (size_t h = 0; h < nirrep_; ++h) {
        auto ncol = T->coldim(h);
        auto nrow = T->rowdim(h);
        for (int q = 0; q < ncol; ++q) {
            double max = 0.0, sign = 1.0;
            int p_temp = q;

            for (int p = 0; p < nrow; ++p) {
                double v = Smo->get(h, p, q);
                if (std::fabs(v) > max) {
                    max = std::fabs(v);
                    p_temp = p;
                    sign = v < 0 ? -1.0 : 1.0;
                }
            }

            T->set(h, p_temp, q, sign);
        }
    }

    // test transformation matrix
    bool trans_ok = true;
    for (size_t h = 0; h < nirrep_; ++h) {
        auto nrow = T->rowdim(h);
        auto ncol = T->coldim(h);

        for (int i = 0; i < nrow; ++i) {
            double sum = 0.0;
            for (int j = 0; j < ncol; ++j) {
                sum += std::fabs(T->get(h, i, j));
            }
            if (sum - 1.0 > 1.0e-3) {
                trans_ok = false;
                break;
            }
        }

        if (not trans_ok) {
            break;
        }
    }

    // transform Ua
    if (trans_ok) {
        Ua = psi::linalg::doublet(Ua, T, false, false);
    } else {
        psi::outfile->Printf("\n  Failed to fix orbital phase and order.\n");
        //        Ca->print();
        //        Cnew->print();
        Smo->print();
        T->print();
    }
}
} // namespace forte
