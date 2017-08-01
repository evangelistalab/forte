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

#include <cmath>

#include "ambit/blocked_tensor.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "../blockedtensorfactory.h"
#include "helpers.h"
#include "semi_canonicalize.h"

namespace psi {
namespace forte {

using namespace ambit;

SemiCanonical::SemiCanonical(std::shared_ptr<Wavefunction> wfn,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info, const bool& quiet)
    : wfn_(wfn), mo_space_info_(mo_space_info), ints_(ints), quiet_(quiet) {

    if (!quiet) {
        print_method_banner(
            {"Semi-Canonical Orbitals", "Jeffrey B. Schriber and Francesco A. Evangelista"});
    }

    // 0. initialize the dimension objects
    nirrep_ = wfn_->nirrep();
    ncmo_ = mo_space_info_->size("CORRELATED");
    nact_ = mo_space_info_->size("ACTIVE");
    nmopi_ = wfn_->nmopi();
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    fdocc_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    rdocc_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->get_dimension("ACTIVE");
    ruocc_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // figure out dimension map
    mo_dims_["core"] = rdocc_;
    mo_dims_["actv"] = actv_;
    mo_dims_["virt"] = ruocc_;

    // figure out index map
    cmo_idx_["core"] = idx_space(rdocc_, Dimension(std::vector<int>(nirrep_, 0)), ncmopi_);
    cmo_idx_["actv"] = idx_space(actv_, rdocc_, ncmopi_);
    cmo_idx_["virt"] = idx_space(ruocc_, rdocc_ + actv_, ncmopi_);

    // figure out offsets
    offsets_["core"] = fdocc_;
    offsets_["actv"] = fdocc_ + rdocc_;
    offsets_["virt"] = fdocc_ + rdocc_ + actv_;

    std::vector<int> actv_off;
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        actv_off.emplace_back(offset);
        offset += actv_[h];
    }
    actv_offsets_["actv"] = actv_off;
}

SemiCanonical::SemiCanonical(std::shared_ptr<Wavefunction> wfn,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info, Dimension actv_docc,
                             Dimension actv_virt, const bool& quiet)
    : wfn_(wfn), mo_space_info_(mo_space_info), ints_(ints), actv_docc_(actv_docc),
      actv_virt_(actv_virt), quiet_(quiet) {

    if (!quiet) {
        print_method_banner(
            {"Semi-Canonical Orbitals", "Jeffrey B. Schriber and Francesco A. Evangelista"});
    }

    // initialization
    nirrep_ = wfn_->nirrep();
    ncmo_ = mo_space_info_->size("CORRELATED");
    nact_ = mo_space_info_->size("ACTIVE");
    nmopi_ = wfn_->nmopi();
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    fdocc_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    rdocc_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->get_dimension("ACTIVE");
    ruocc_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // test actv_docc and actv_virt
    Dimension actv = actv_docc + actv_virt;
    if (actv != actv_) {
        throw PSIEXCEPTION("ACTIVE_DOCC and ACTIVE_VIRT do not add up to ACTIVE!");
    }

    // figure out dimension map
    mo_dims_["core"] = rdocc_;
    mo_dims_["actv_docc"] = actv_docc_;
    mo_dims_["actv_virt"] = actv_virt_;
    mo_dims_["virt"] = ruocc_;

    // figure out index map
    cmo_idx_["core"] = idx_space(rdocc_, Dimension(std::vector<int>(nirrep_, 0)), ncmopi_);
    cmo_idx_["actv_docc"] = idx_space(actv_docc_, rdocc_, ncmopi_);
    cmo_idx_["actv_virt"] = idx_space(actv_virt_, rdocc_ + actv_docc, ncmopi_);
    cmo_idx_["virt"] = idx_space(ruocc_, rdocc_ + actv_, ncmopi_);

    // figure out offsets
    offsets_["core"] = fdocc_;
    offsets_["actv_docc"] = fdocc_ + rdocc_;
    offsets_["actv_virt"] = fdocc_ + rdocc_ + actv_docc_;
    offsets_["virt"] = fdocc_ + rdocc_ + actv_;

    std::vector<int> actvh_off, actvp_off;
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        actvh_off.emplace_back(offset);
        offset += actv_docc[h];
        actvp_off.emplace_back(offset);
        offset += actv_virt[h];
    }
    actv_offsets_["actv_docc"] = actvh_off;
    actv_offsets_["actv_virt"] = actvp_off;
}

std::vector<std::vector<size_t>>
SemiCanonical::idx_space(const Dimension& npi, const Dimension& bpi, const Dimension& tpi) {
    std::vector<std::vector<size_t>> out(nirrep_, std::vector<size_t>());

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        offset += bpi[h];
        for (int i = 0; i < npi[h]; ++i) {
            out[h].emplace_back(offset + i);
        }
        offset += tpi[h] - bpi[h];
    }

    return out;
}

void SemiCanonical::semicanonicalize(Reference& reference) {
    Timer SemiCanonicalize;

    // 1. Build the Fock matrix
    build_fock_matrix(reference);

    // Check Fock matrix
    check_fock_matrix();

    bool semi = true;
    for (const auto& name_semi_pair : checked_results_) {
        if (name_semi_pair.second) {
            semi = false;
        }
    }

    if (semi) {
        outfile->Printf("\n  Orbitals are already semicanonicalized.");
    } else {
        // 2. Build transformation matrices from diagononalizing blocks in F

        // These transform all MOs
        SharedMatrix Ua(new Matrix("Ua", nmopi_, nmopi_));
        SharedMatrix Ub(new Matrix("Ub", nmopi_, nmopi_));

        // This transforms only within ACTIVE MOs
        // Use ambit Tensor here so that the ambit mo_spaces remains the same
        ambit::Tensor Ua_t = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
        ambit::Tensor Ub_t = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

        build_transformation_matrices(Ua, Ub, Ua_t, Ub_t);

        // 3. Retransform integrals
        transform_ints(Ua, Ub);

        // 4. Transform RMDs/cumulants
        transform_reference(Ua_t, Ub_t, reference);

        outfile->Printf("\n  SemiCanonicalize takes %8.6f s.", SemiCanonicalize.get());
    }
}

void SemiCanonical::build_fock_matrix(Reference& reference) {
    // 1. Build the Fock matrix

    SharedMatrix Da(new Matrix("Da", ncmo_, ncmo_));
    SharedMatrix Db(new Matrix("Db", ncmo_, ncmo_));

    Matrix L1a = tensor_to_matrix(reference.L1a(), actv_);
    Matrix L1b = tensor_to_matrix(reference.L1b(), actv_);

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        // core block (diagonal)
        for (int i = 0; i < rdocc_[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Da->set(offset + u, offset + v, L1a.get(h, u, v));
                Db->set(offset + u, offset + v, L1b.get(h, u, v));
            }
        }

        offset += ncmopi_[h] - rdocc_[h];
    }

    Timer FockTime;
    ints_->make_fock_matrix(Da, Db);
    outfile->Printf("\n  Took %8.6f s to build Fock matrix", FockTime.get());
}

void SemiCanonical::check_fock_matrix() {
    print_h2("Checking Fock Matrix Diagonal Blocks");
    int width = 12 + 2 + 13 + 2 + 13;
    std::string dash(width, '-');
    outfile->Printf("\n    %12c  %5c%s%5c  %4c%s", ' ', ' ', "Max", ' ', ' ', "2-Norm");
    outfile->Printf("\n    %s", dash.c_str());

    // universial threshold
    double e_conv = (wfn_->options()).get_double("E_CONVERGENCE");
    double threshold_max = 10.0 * e_conv;

    // loop over orbital spaces
    for (const auto& name_dim_pair : mo_dims_) {
        std::string name = name_dim_pair.first;
        std::string name_a = "Fa " + name;
        std::string name_b = "Fb " + name;
        Dimension npi = name_dim_pair.second;

        // build Fock matrix of this diagonal block
        SharedMatrix Fa(new Matrix(name_a, npi, npi));
        SharedMatrix Fb(new Matrix(name_b, npi, npi));

        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < npi[h]; ++i) {
                for (int j = 0; j < npi[h]; ++j) {
                    Fa->set(h, i, j, ints_->fock_a(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                    Fb->set(h, i, j, ints_->fock_b(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
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
        outfile->Printf("\n    %-12s  %13.10f  %13.10f", name_a.c_str(), Famax, Fanorm);
        outfile->Printf("\n    %-12s  %13.10f  %13.10f", name_b.c_str(), Fbmax, Fbnorm);
        outfile->Printf("\n    %s", dash.c_str());

        // check threshold
        double threshold_norm = npi.sum() * (npi.sum() - 1) * e_conv;
        bool FaDo = (Famax <= threshold_max && Fanorm <= threshold_norm) ? false : true;
        bool FbDo = (Fbmax <= threshold_max && Fbnorm <= threshold_norm) ? false : true;
        checked_results_[name] = FaDo && FbDo;
    }
}

void SemiCanonical::build_transformation_matrices(SharedMatrix& Ua, SharedMatrix& Ub,
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
        Dimension npi = name_dim_pair.second;
        bool FockDo = checked_results_[name];

        if (FockDo) {
            // build Fock matrix of this diagonal block
            SharedMatrix Fa(new Matrix(name_a, npi, npi));
            SharedMatrix Fb(new Matrix(name_b, npi, npi));

            for (int h = 0; h < nirrep_; ++h) {
                for (int i = 0; i < npi[h]; ++i) {
                    for (int j = 0; j < npi[h]; ++j) {
                        Fa->set(h, i, j, ints_->fock_a(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                        Fb->set(h, i, j, ints_->fock_b(cmo_idx_[name][h][i], cmo_idx_[name][h][j]));
                    }
                }
            }

            // diagonalize this Fock block
            SharedMatrix UsubA(new Matrix("Ua " + name, npi, npi));
            SharedMatrix UsubB(new Matrix("Ub " + name, npi, npi));
            SharedVector evalsA(new Vector("evals a " + name, npi));
            SharedVector evalsB(new Vector("evals b " + name, npi));
            Fa->diagonalize(UsubA, evalsA);
            Fb->diagonalize(UsubB, evalsB);

            // fill in Ua and Ub
            for (int h = 0; h < nirrep_; ++h) {
                int offset = offsets_[name][h];
                for (int i = 0; i < npi[h]; ++i) {
                    for (int j = 0; j < npi[h]; ++j) {
                        Ua->set(h, offset + i, offset + j, UsubA->get(h, i, j));
                        Ub->set(h, offset + i, offset + j, UsubB->get(h, i, j));
                    }
                }
            }

            // fill in UaData and UbData if this block is active
            if (name.find("actv") != std::string::npos) {
                for (int h = 0; h < nirrep_; ++h) {
                    int actv_off = actv_offsets_[name][h];
                    for (int u = 0; u < npi[h]; ++u) {
                        for (int v = 0; v < npi[h]; ++v) {
                            int nu = actv_off + u;
                            int nv = actv_off + v;
                            UaData[nu * nact_ + nv] = UsubA->get(h, u, v);
                            UbData[nu * nact_ + nv] = UsubB->get(h, u, v);
                        }
                    }
                }
            }
        }
    }

    // copy active data to ambit tensors
    Ua_t.data() = UaData;
    Ub_t.data() = UbData;
}

void SemiCanonical::transform_ints(SharedMatrix& Ua, SharedMatrix& Ub) {
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Cb = wfn_->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false, false, 1.0, Ca, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, Ub, 0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // 5. Retransform the integrals in the new basis
    print_h2("Integral Transformation");
    ints_->retransform_integrals();
}

void SemiCanonical::transform_reference(ambit::Tensor& Ua, ambit::Tensor& Ub,
                                        Reference& reference) {
    print_h2("Reference Transformation");

    // Transform the 1-cumulants
    ambit::Tensor L1a0 = reference.L1a();
    ambit::Tensor L1b0 = reference.L1b();

    ambit::Tensor L1aT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1a", {nact_, nact_});
    ambit::Tensor L1bT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1b", {nact_, nact_});
    L1aT("pq") = Ua("ap") * L1a0("ab") * Ua("bq");
    L1bT("PQ") = Ub("AP") * L1b0("AB") * Ub("BQ");

    reference.set_L1a(L1aT);
    reference.set_L1b(L1bT);
    outfile->Printf("\n    Transformed 1 cumulants.");

    // Transform 2-cumulants and recompute 2-RDMs using transformed cumulants
    ambit::Tensor L2aa0 = reference.L2aa();
    ambit::Tensor L2ab0 = reference.L2ab();
    ambit::Tensor L2bb0 = reference.L2bb();

    //   aa spin
    ambit::Tensor L2T =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2aa", {nact_, nact_, nact_, nact_});
    L2T("pqrs") = Ua("ap") * Ua("bq") * L2aa0("abcd") * Ua("cr") * Ua("ds");
    L2aa0.copy(L2T);

    ambit::Tensor G2T =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed G2aa", {nact_, nact_, nact_, nact_});
    G2T.copy(L2T);
    G2T("pqrs") += L1aT("pr") * L1aT("qs");
    G2T("pqrs") -= L1aT("ps") * L1aT("qr");
    reference.set_g2aa(G2T.clone());

    //   ab spin
    L2T.set_name("Transformed L2ab");
    L2T("pQrS") = Ua("ap") * Ub("BQ") * L2ab0("aBcD") * Ua("cr") * Ub("DS");
    L2ab0.copy(L2T);

    G2T.set_name("Transformed G2ab");
    G2T.copy(L2T);
    G2T("pqrs") += L1aT("pr") * L1bT("qs");
    reference.set_g2ab(G2T.clone());

    //   bb spin
    L2T.set_name("Transformed L2bb");
    L2T("PQRS") = Ub("AP") * Ub("BQ") * L2bb0("ABCD") * Ub("CR") * Ub("DS");
    L2bb0.copy(L2T);

    G2T.set_name("Transformed G2bb");
    G2T.copy(L2T);
    G2T("pqrs") += L1bT("pr") * L1bT("qs");
    G2T("pqrs") -= L1bT("ps") * L1bT("qr");
    reference.set_g2bb(G2T.clone());

    outfile->Printf("\n    Transformed 2 cumulants and RDMs.");

    // Transform 3 cumulants
    ambit::Tensor L3aaa0 = reference.L3aaa();
    ambit::Tensor L3aab0 = reference.L3aab();
    ambit::Tensor L3abb0 = reference.L3abb();
    ambit::Tensor L3bbb0 = reference.L3bbb();

    ambit::Tensor L3T =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L3aaa", std::vector<size_t>(6, nact_));
    L3T("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * L3aaa0("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    L3aaa0.copy(L3T);

    L3T.set_name("Transformed L3aab");
    L3T("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * L3aab0("abCijK") * Ua("is") * Ua("jt") * Ub("KU");
    L3aab0.copy(L3T);

    L3T.set_name("Transformed L3abb");
    L3T("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * L3abb0("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");
    L3abb0.copy(L3T);

    L3T.set_name("Transformed L3bbb");
    L3T("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * L3bbb0("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");
    L3bbb0.copy(L3T);

    outfile->Printf("\n    Transformed 3 cumulants.");
}
}
} // End Namespaces
