/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/sieve.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include "LaplaceDenominator.h"
#include "psi4/libfock/jk.h"
#include "Laplace.h"
#include "ao_helper.h"

#include <numeric> 

using namespace psi;

namespace forte {

AtomicOrbitalHelper::AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ,
                                         psi::SharedVector eps_vir, double laplace_tolerance)
    : CMO_(CMO), eps_rdocc_(eps_occ), eps_virtual_(eps_vir), laplace_tolerance_(laplace_tolerance) {
    LaplaceDenominator laplace(eps_rdocc_, eps_virtual_, laplace_tolerance_);
    Occupied_Laplace_ = laplace.denominator_occ();
    Virtual_Laplace_ = laplace.denominator_vir();
    weights_ = Occupied_Laplace_->rowspi()[0];
    nrdocc_ = eps_rdocc_->dim();
    nvir_ = eps_virtual_->dim();
    nbf_ = CMO_->rowspi()[0];
    shift_ = 0;
}
AtomicOrbitalHelper::AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ,
                                         psi::SharedVector eps_vir, double laplace_tolerance,
                                         int shift, int nfrozen)
    : CMO_(CMO), eps_rdocc_(eps_occ), eps_virtual_(eps_vir), laplace_tolerance_(laplace_tolerance),
      shift_(shift), nfrozen_(nfrozen) {
    LaplaceDenominator laplace(eps_rdocc_, eps_virtual_, laplace_tolerance_);
    Occupied_Laplace_ = laplace.denominator_occ();
    Virtual_Laplace_ = laplace.denominator_vir();
    weights_ = Occupied_Laplace_->rowspi()[0];
    nrdocc_ = eps_rdocc_->dim();
    nvir_ = eps_virtual_->dim();
    nbf_ = CMO_->rowspi()[0];
    vir_start_ = 0;
}

AtomicOrbitalHelper::AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_act,
                                         psi::SharedVector eps_vir, double laplace_tolerance,
                                         int shift, int nfrozen, bool cavv, double vir_tol)
    : CMO_(CMO), eps_rdocc_(eps_occ), eps_active_(eps_act), eps_virtual_(eps_vir), laplace_tolerance_(laplace_tolerance),
      shift_(shift), nfrozen_(nfrozen) {
    LaplaceDenominator laplace(eps_rdocc_, eps_active_, eps_virtual_, laplace_tolerance_, cavv, vir_tol);
    Occupied_Laplace_ = laplace.denominator_occ();
    Virtual_Laplace_ = laplace.denominator_vir();
    Active_Laplace_ = laplace.denominator_act();
    weights_ = Occupied_Laplace_->rowspi()[0];
    nrdocc_ = eps_rdocc_->dim();
    nact_ = eps_active_->dim();
    nvir_ = Virtual_Laplace_->colspi()[0];
    vir_start_ = laplace.vir_start();
    shift_ += vir_start_;
    nbf_ = CMO_->rowspi()[0];
}

AtomicOrbitalHelper::~AtomicOrbitalHelper() { outfile->Printf("\n    Done with AO helper class"); }

void AtomicOrbitalHelper::Compute_Cholesky_Pseudo_Density() {
    psi::SharedMatrix POcc_single(new psi::Matrix("Single_POcc", nbf_, nbf_));
    psi::SharedMatrix PVir_single(new psi::Matrix("Single_PVir", nbf_, nbf_)); 

    double value_occ, value_vir = 0.0;
    for (int w = 0; w < weights_; w++) {
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                for (int i = 0; i < nrdocc_; i++) {
                    value_occ += CMO_->get(mu, i + nfrozen_) * CMO_->get(nu, i + nfrozen_) * Occupied_Laplace_->get(w, i);
                }
                POcc_single->set(mu, nu, value_occ);
                for (int a = 0; a < nvir_; a++) {
                    value_vir += CMO_->get(mu, nfrozen_ + nrdocc_ + shift_ + a) *
                                 CMO_->get(nu, nfrozen_ + nrdocc_ + shift_ + a) * Virtual_Laplace_->get(w, a);
                }
                PVir_single->set(mu, nu, value_vir);
                value_occ = 0.0;
                value_vir = 0.0;
            }
        }
        psi::SharedMatrix LOcc = POcc_single->partial_cholesky_factorize(1e-10);
        psi::SharedMatrix LVir = PVir_single->partial_cholesky_factorize(1e-10);
        LOcc_list_.push_back(LOcc);
        LVir_list_.push_back(LVir);
        POcc_list_.push_back(POcc_single);
        PVir_list_.push_back(PVir_single);
        n_pseudo_occ_list_.push_back(LOcc->coldim());
        n_pseudo_vir_list_.push_back(LVir->coldim());
    }
}

void AtomicOrbitalHelper::Compute_Cholesky_Active_Density(psi::SharedMatrix RDM) {
    psi::SharedMatrix PAct_single(new psi::Matrix("Single_PAct", nbf_, nbf_));
    std::vector<int> Act_idx(nact_);
    std::iota(Act_idx.begin(), Act_idx.end(), nfrozen_+nrdocc_);
    psi::SharedMatrix C_Act = submatrix_cols(*CMO_, Act_idx);
    psi::SharedMatrix D_Act = psi::linalg::doublet(C_Act, RDM, false, false);

    double value_act = 0.0;
    for (int w = 0; w < weights_; w++) {
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                for (int u = 0; u < nact_; u++) {
                    value_act += D_Act->get(mu, u) * C_Act->get(nu, u) * Active_Laplace_->get(w, u);
                }
                PAct_single->set(mu, nu, value_act);
                value_act = 0.0;
            }
        }
        psi::SharedMatrix LAct = PAct_single->partial_cholesky_factorize(1e-10);
        LAct_list_.push_back(LAct);
        n_pseudo_vir_list_.push_back(LAct->coldim());
    }
}

void AtomicOrbitalHelper::Compute_Cholesky_Density() {
    std::vector<int> Occ_idx(nrdocc_);
    std::iota(Occ_idx.begin(), Occ_idx.end(), nfrozen_);
    psi::SharedMatrix C_Occ = submatrix_cols(*CMO_, Occ_idx);
    POcc_real_ = psi::linalg::doublet(C_Occ, C_Occ, false, true);
    L_Occ_real_ = POcc_real_->partial_cholesky_factorize(1e-10);
}
/*
void AtomicOrbitalHelper::Householder_QR() {
    std::vector<int> Occ_idx(nrdocc_);
    std::iota(Occ_idx.begin(), Occ_idx.end(), 0);
    psi::SharedMatrix C_Occ = submatrix_cols(*CMO_, Occ_idx);
    psi::SharedMatrix C_Occ_transpose = C_Occ->transpose();
    /// Copy C_Occ matrix to R
    double** C_T = C_Occ_transpose->pointer();
    auto R = std::make_shared<psi::Matrix>("R", nrdocc_, C_Occ_transpose->coldim());
    double** Rp = R->pointer();
    C_DCOPY(nrdocc_ * C_Occ_transpose->coldim(), C_T[0], 1, Rp[0], 1);

    /// QR decomposition
    std::vector<double> tau(nrdocc_);
    //std::vector<int> jpvt(C_Occ_transpose->coldim());

    // First, find out how much workspace to provide
    // Optimal size of work vector is written to work_size
    double work_size;
    //C_DGEQP3(nrdocc_, C_Occ_transpose->coldim(), Rp[0], nrdocc_, jpvt.data(), tau.data(), &work_size, -1);
    C_DGEQRF(nrdocc_, C_Occ_transpose->coldim(), Rp[0], nrdocc_, tau.data(), &work_size, -1);
    // Now, do the QR decomposition
    int lwork = (int)work_size;
    std::vector<double> work(lwork);
    //C_DGEQP3(nrdocc_, C_Occ_transpose->coldim(), Rp[0], nrdocc_, jpvt.data(), tau.data(), work.data(), lwork);
    C_DGEQRF(nrdocc_, C_Occ_transpose->coldim(), Rp[0], nrdocc_, tau.data(), work.data(), lwork);

    // Put R in the upper triangle where it belongs
    for (int i = 1; i < nrdocc_; i++) {
        for (int j = 0; j < i; j++) {
            Rp[j][i] = 0.0;
        }
    }

    R->transpose()->print();
}
*/
} // namespace forte