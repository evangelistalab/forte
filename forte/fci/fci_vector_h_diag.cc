/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"

#include "helpers/timer.h"
#include "integrals/active_space_integrals.h"
#include "fci_vector.h"
#include "binary_graph.hpp"
#include "string_lists.h"
#include "string_address.h"

namespace forte {

void FCIVector::form_H_diagonal(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    local_timer t;

    Determinant I;
    // loop over all irreps of the alpha strings
    for (int ha = 0; ha < nirrep_; ha++) {
        const int hb = ha ^ symmetry_;
        double** C_ha = C_[ha]->pointer();
        const auto& sa = lists_->alfa_strings()[ha];
        const auto& sb = lists_->beta_strings()[hb];
        for (const auto& Ia : sa) {
            for (const auto& Ib : sb) {
                size_t addIa = alfa_address_->add(Ia);
                size_t addIb = beta_address_->add(Ib);
                I.set_str(Ia, Ib);
                C_ha[addIa][addIb] = fci_ints->energy(I) + fci_ints->scalar_energy() +
                                     fci_ints->nuclear_repulsion_energy();
            }
        }
    }

    hdiag_timer += t.get();
    if (print_) {
        psi::outfile->Printf("\n  Timing for Hdiag          = %10.3f s", hdiag_timer);
    }
}

std::vector<std::tuple<double, size_t, size_t, size_t>> FCIVector::min_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    double emax = std::numeric_limits<double>::max();
    size_t added = 0;

    std::vector<std::tuple<double, size_t, size_t, size_t>> dets(num_dets);
    for (auto& det : dets) {
        std::get<0>(det) = emax;
    }

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpi(alfa_sym);
        size_t maxIb = beta_address_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                double e = C_ha[Ia][Ib];
                if ((e < emax) or (added < num_dets)) {
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it =
                        std::find_if(dets.begin(), dets.end(),
                                     [&e](const std::tuple<double, size_t, size_t, size_t>& t) {
                                         return e < std::get<0>(t);
                                     });
                    dets.insert(it, std::make_tuple(e, alfa_sym, Ia, Ib));
                    emax = std::get<0>(dets.back());
                    added++;
                }
            }
        }
    }
    return dets;
}

std::vector<std::tuple<double, double, size_t, size_t, size_t>>
FCIVector::max_abs_elements(size_t num_dets) {
    num_dets = std::min(num_dets, ndet_);

    std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets(num_dets);

    double emin = 0.0;

    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpi(alfa_sym);
        size_t maxIb = beta_address_->strpi(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                double e = std::fabs(C_ha[Ia][Ib]);
                if (e > emin) {
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it = std::find_if(
                        dets.begin(), dets.end(),
                        [&e](const std::tuple<double, double, size_t, size_t, size_t>& t) {
                            return e > std::get<0>(t);
                        });
                    dets.insert(it, std::make_tuple(e, C_ha[Ia][Ib], alfa_sym, Ia, Ib));
                    emin = std::get<0>(dets.back());
                }
            }
        }
    }
    return dets;
}

} // namespace forte
