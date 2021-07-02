/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/lib3index/denominator.h"
#include "psi4/libfock/jk.h"
#include "ao_helper.h"

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
void AtomicOrbitalHelper::Compute_Psuedo_Density() {
    int nmo_ = nbf_;
    psi::SharedMatrix Xocc(new psi::Matrix("DensityOccupied", weights_, nbf_ * nbf_));
    psi::SharedMatrix Yvir(new psi::Matrix("DensityVirtual", weights_, nmo_ * nmo_));

    double value_occ, value_vir = 0;
    for (int w = 0; w < weights_; w++) {
        for (int mu = 0; mu < nbf_; mu++) {
            for (int nu = 0; nu < nbf_; nu++) {
                for (int i = 0; i < nrdocc_; i++) {
                    value_occ += CMO_->get(mu, i) * CMO_->get(nu, i) * Occupied_Laplace_->get(w, i);
                }
                Xocc->set(w, mu * nmo_ + nu, value_occ);
                for (int a = 0; a < nvir_; a++) {
                    value_vir += CMO_->get(mu, nrdocc_ + shift_ + a) *
                                 CMO_->get(nu, nrdocc_ + shift_ + a) * Virtual_Laplace_->get(w, a);
                }
                Yvir->set(w, mu * nmo_ + nu, value_vir);
                value_occ = 0.0;
                value_vir = 0.0;
            }
        }
    }
    POcc_ = Xocc->clone();
    PVir_ = Yvir->clone();
}
void AtomicOrbitalHelper::Compute_AO_Screen(std::shared_ptr<psi::BasisSet>& primary) {
    ERISieve sieve(primary, 1e-10);
    std::vector<double> my_function_pair_values = sieve.function_pair_values();
    psi::SharedMatrix AO_Screen(new psi::Matrix("Z", nbf_, nbf_));
    for (int mu = 0; mu < nbf_; mu++)
        for (int nu = 0; nu < nbf_; nu++)
            AO_Screen->set(mu, nu, my_function_pair_values[mu * nbf_ + nu]);

    AO_Screen_ = AO_Screen;
    AO_Screen_->set_name("ScwartzAOInts");
}
void AtomicOrbitalHelper::Estimate_TransAO_Screen(std::shared_ptr<psi::BasisSet>& primary,
                                                  std::shared_ptr<psi::BasisSet>& auxiliary) {
    Compute_Psuedo_Density();
    std::shared_ptr<JK> jk(new MemDFJK(primary, auxiliary));
    jk->initialize();
    jk->compute();
    psi::SharedMatrix AO_Trans_Screen(new psi::Matrix("AOTrans", weights_, nbf_ * nbf_));

    for (int w = 0; w < weights_; w++) {
        psi::SharedMatrix COcc(new psi::Matrix("COcc", nbf_, nbf_));
        psi::SharedMatrix CVir(new psi::Matrix("COcc", nbf_, nbf_));
        for (int mu = 0; mu < nbf_; mu++)
            for (int nu = 0; nu < nbf_; nu++) {
                COcc->set(mu, nu, POcc_->get(w, mu * nbf_ + nu));
                CVir->set(mu, nu, PVir_->get(w, mu * nbf_ + nu));
            }

        psi::SharedVector iaia_w = jk->iaia(COcc, CVir);
        for (int mu = 0; mu < nbf_; mu++)
            for (int nu = 0; nu < nbf_; nu++)
                AO_Trans_Screen->set(w, mu * nbf_ + nu, iaia_w->get(mu * nbf_ + nu));
    }
    TransAO_Screen_ = AO_Trans_Screen;
    TransAO_Screen_->set_name("(u_b {b}^v | u_b {b}^v)");
}
} // namespace forte
