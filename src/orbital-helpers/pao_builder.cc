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
#include <vector>

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/integral.h"
#include "psi4/libpsio/psio.hpp"

#include "pao_builder.h"

using namespace psi;

namespace forte {

PAObuilder::PAObuilder(const psi::SharedMatrix C, psi::Dimension noccpi,
                       std::shared_ptr<BasisSet> basis)
    : C_(C), noccpi_(noccpi), basis_(basis) {
    startup();
}

void PAObuilder::startup() {
    // Read info
    nirrep_ = noccpi_.n();
    nbf_ = basis_->nbf();
    nmopi_ = noccpi_; // This will keep other sym unchange
    nmopi_[0] = nbf_; // Assume all virtual to be A1

    // Build D
    outfile->Printf("\n ****** Build Density ******");

    SharedMatrix D(new Matrix("Density pq", nirrep_, nmopi_, nmopi_));
    for (int h = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nmopi_[h]; ++u) {
            for (int v = 0; v < nmopi_[h]; ++v) {
                double d_value = 0.0;
                for (int i = 0; i < noccpi_[h]; ++i) {
                    d_value += C_->get(h, u, i) * C_->get(h, v, i);
                }
                D->set(h, u, v, d_value);
            }
        }
    }
    D_ = D;

    // Build S
    outfile->Printf("\n ****** Build overlap ******");
    std::shared_ptr<IntegralFactory> integral_pp(
        new IntegralFactory(basis_, basis_, basis_, basis_));
    std::shared_ptr<OneBodyAOInt> S_int(integral_pp->ao_overlap());

    SharedMatrix S_nn = std::make_shared<psi::Matrix>("S_nn", nbf_, nbf_);
    S_int->compute(S_nn);
    S_ = S_nn;

    outfile->Printf("\n ****** Initialization done! ******");
}

SharedMatrix PAObuilder::build_A_virtual(int nbf_A, double pao_threshold) {
    // Build fragment virtual
    outfile->Printf("\n ****** Build Slices ******");
    Dimension nbfA = nmopi_; // Just in case if we want to use PAO with sym in the future
    nbfA[0] = nbf_A;         // All built PAOs will be assumed A1
    Dimension zeropi = nmopi_ - nmopi_;

    Slice A(zeropi, nbfA);
    Slice AB(zeropi, nmopi_);

    outfile->Printf("\n ****** Compute C_pao ******");
    SharedMatrix I_aa(new Matrix("Identity with A*A size", nirrep_, nbfA, nbfA));
    I_aa->identity();

    SharedMatrix S_na = S_->get_block(AB, A);
    SharedMatrix C_pao(new Matrix("C_pao, with N*A size", nirrep_, nmopi_, nbfA));
    C_pao->set_block(A, A, I_aa);

    // Build C_pao = I - DS
    C_pao->subtract(linalg::doublet(D_, S_na));

    outfile->Printf("\n ****** Orthogonalize C_pao ******");
    // Orthogonalize C_pao
    SharedMatrix U(new Matrix("U", nirrep_, nbfA, nbfA));
    SharedVector lambda(new Vector("lambda", nirrep_, nbfA));
    SharedMatrix S_pao_A = linalg::triplet(C_pao, S_, C_pao, true, false, false);
    S_pao_A->diagonalize(U, lambda, descending);

    // Truncate A_virtual and build s_-1/2
    int num_pao_A = 0;
    int num_pao_B = 0;

    outfile->Printf("\n PAO truncation: %8.6f", pao_threshold);

    for (int i = 0; i < nbf_A; ++i) {
        double leig = lambda->get(0, i);
        if (leig > pao_threshold) {
            ++num_pao_A;
            lambda->set(0, i, 1.0 / sqrt(leig));
            outfile->Printf("\n PAO %d: %8.8f", i, leig);
        } else {
            ++num_pao_B;
        }
    }

    Dimension VA_short = nmopi_;
    VA_short[0] = num_pao_A;
    Slice Ashort(zeropi, VA_short);

    // Compute C_opao
    SharedMatrix CU = linalg::doublet(C_pao, U);
    SharedMatrix L(new Matrix("Lambda^-1/2, with Ashort*Ashort size", nirrep_, VA_short, VA_short));
    for (int i = 0; i < num_pao_A; ++i) {
        L->set(0, i, i, lambda->get(0, i));
    }

    SharedMatrix C_short = linalg::doublet(CU->get_block(AB, Ashort), L->get_block(Ashort, Ashort));
    outfile->Printf("\n ****** PAOs generated ******");
    return C_short;
}

SharedMatrix PAObuilder::build_B_virtual() {
    // Build environment virtual
    SharedMatrix C_virtual_B(new Matrix("C_vir B", nirrep_, nmopi_, nmopi_));
    throw PSIEXCEPTION("Environment PAO generations not available now!");
    return C_virtual_B;
}

} // namespace forte
