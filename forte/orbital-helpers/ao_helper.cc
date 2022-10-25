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

#include "psi4/lib3index/denominator.h"
#include "psi4/libfock/jk.h"
#include "Laplace.h"
#include "ao_helper.h"

#include <numeric> 

using namespace psi;

namespace forte {

AtomicOrbitalHelper::AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ,
                                         psi::SharedVector eps_vir, double laplace_tolerance)
    : CMO_(CMO), eps_rdocc_(eps_occ), eps_virtual_(eps_vir), laplace_tolerance_(laplace_tolerance) {
    psi::LaplaceDenominator laplace(eps_rdocc_, eps_virtual_, laplace_tolerance_);
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
                                         int shift)
    : CMO_(CMO), eps_rdocc_(eps_occ), eps_virtual_(eps_vir), laplace_tolerance_(laplace_tolerance),
      shift_(shift) {
    psi::LaplaceDenominator laplace(eps_rdocc_, eps_virtual_, laplace_tolerance_);
    Occupied_Laplace_ = laplace.denominator_occ();
    Virtual_Laplace_ = laplace.denominator_vir();
    weights_ = Occupied_Laplace_->rowspi()[0];
    nrdocc_ = eps_rdocc_->dim();
    nvir_ = eps_virtual_->dim();
    nbf_ = CMO_->rowspi()[0];
}
AtomicOrbitalHelper::~AtomicOrbitalHelper() { outfile->Printf("\n Done with AO helper class"); }
/*
void AtomicOrbitalHelper::Compute_L_Directly() {
    int nmo_ = nbf_;

    LOcc_list_.resize(weights_);
    LVir_list_.resize(weights_);
    double value_occ, value_vir = 0;
    for (int w = 0; w < weights_; w++) {
        LOcc_list_[w] = std::make_shared<Matrix>("LOcc_list", nbf_, nrdocc_);
        LVir_list_[w] = std::make_shared<Matrix>("LVir_list", nbf_, nvir_);
        for (int mu = 0; mu < nbf_; mu++) {
            for (int i = 0; i < nrdocc_; i++) {
                value_occ = CMO_->get(mu, i) * std::sqrt(Occupied_Laplace_->get(w, i));
                LOcc_list_[w]->set(mu, i, value_occ);
                value_occ = 0.0;
            }
            for (int a = 0; a < nvir_; a++) {
                value_vir = CMO_->get(mu, nrdocc_ + shift_ + a) * std::sqrt(Virtual_Laplace_->get(w, a));
                LVir_list_[w]->set(mu, a, value_vir);
                value_vir = 0.0;
            }
        }
        n_pseudo_occ_list_.push_back(LOcc_list_[w]->coldim());
        n_pseudo_vir_list_.push_back(LVir_list_[w]->coldim());
    }
}
*/
void AtomicOrbitalHelper::Compute_Cholesky_Pseudo_Density() {
    psi::SharedMatrix POcc_single(new psi::Matrix("Single_POcc", nbf_, nbf_));
    psi::SharedMatrix PVir_single(new psi::Matrix("Single_PVir", nbf_, nbf_)); 

    double value_occ, value_vir = 0.0;
    for (int w = 0; w < weights_; w++) {
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                for (int i = 0; i < nrdocc_; i++) {
                    value_occ += CMO_->get(mu, i) * CMO_->get(nu, i) * Occupied_Laplace_->get(w, i);
                }
                POcc_single->set(mu, nu, value_occ);
                for (int a = 0; a < nvir_; a++) {
                    value_vir += CMO_->get(mu, nrdocc_ + shift_ + a) *
                                 CMO_->get(nu, nrdocc_ + shift_ + a) * Virtual_Laplace_->get(w, a);
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
void AtomicOrbitalHelper::Compute_Cholesky_Density() {
    psi::SharedMatrix POcc_real(new psi::Matrix("Real_POcc", nbf_, nbf_));
    psi::SharedMatrix PVir_real(new psi::Matrix("Real_PVir", nbf_, nbf_));
    std::vector<int> Occ_idx(nrdocc_);
    std::iota(Occ_idx.begin(), Occ_idx.end(), 0);
    std::vector<int> Vir_idx(nvir_);
    std::iota(Vir_idx.begin(), Vir_idx.end(), nrdocc_ + shift_);
    psi::SharedMatrix C_Occ = submatrix_cols(*CMO_, Occ_idx);
    psi::SharedMatrix C_Vir = submatrix_cols(*CMO_, Vir_idx);

    POcc_real = psi::linalg::doublet(C_Occ, C_Occ, false, true);
    PVir_real = psi::linalg::doublet(C_Vir, C_Vir, false, true);

    L_Occ_real_ = POcc_real->partial_cholesky_factorize(1e-10);
    L_Vir_real_ = PVir_real->partial_cholesky_factorize(1e-10);
}
} // namespace forte