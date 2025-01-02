/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "fci_string_lists.h"
#include "fci_string_address.h"

#include "fci_vector.h"

namespace forte {

double FCIVector::compute_spin2() {
    double spin2 = 0.0;
    // Loop over blocks of matrix C
    for (int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym) {
        const int Ib_sym = Ia_sym ^ symmetry_;
        auto Cr = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for (int rs_sym = 0; rs_sym < nirrep_; ++rs_sym) {
            const int Jb_sym = Ib_sym ^ rs_sym;
            const int Ja_sym = Jb_sym ^ symmetry_;
            auto Cl = C_[Ja_sym]->pointer();
            for (int r_sym = 0; r_sym < nirrep_; ++r_sym) {
                int s_sym = rs_sym ^ r_sym;

                for (int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel) {
                    const int r_abs = r_rel + cmopi_offset_[r_sym];
                    for (int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel) {
                        const int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,Ib_sym)
                        const auto& vo_alfa = lists_->get_alfa_vo_list(s_abs, r_abs, Ia_sym);
                        const auto& vo_beta = lists_->get_beta_vo_list(r_abs, s_abs, Ib_sym);

                        for (const auto& [sign_a, Ia, Ja] : vo_alfa) {
                            for (const auto& [sign_b, Ib, Jb] : vo_beta) {
                                spin2 += Cl[Ja][Jb] * Cr[Ia][Ib] * sign_a * sign_b;
                            }
                        }
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
    double na = alfa_address_->nones();
    double nb = beta_address_->nones();
    return -spin2 + 0.25 * std::pow(na - nb, 2.0) + 0.5 * (na + nb);
}

} // namespace forte
