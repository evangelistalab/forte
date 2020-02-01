/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
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

#include <algorithm>

#include "sparse_ci/determinant.h"
#include "integrals/active_space_integrals.h"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

double slater_rules_single_alpha(String Ib, String Ia, String Ja,
                                 const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    size_t N = Determinant::nbits_half;
    String IJa = Ia ^ Ja;
    uint64_t i = IJa.find_and_clear_first_one();
    uint64_t a = IJa.find_first_one();

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    for (int p = 0; p < N; ++p) {
        if (Ia.get_bit(p)) {
            matrix_element += ints->tei_aa(i, p, a, p);
        }
        if (Ib.get_bit(p)) {
            matrix_element += ints->tei_ab(i, p, a, p);
        }
    }
    return (Ia.slater_sign(i, a) * matrix_element);
}

double slater_rules_single_beta(String Ia, String Ib, String Jb,
                                const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    size_t N = Determinant::nbits_half;
    String IJb = Ib ^ Jb;
    uint64_t i = IJb.find_and_clear_first_one();
    uint64_t a = IJb.find_first_one();

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    for (int p = 0; p < N; ++p) {
        if (Ia.get_bit(p)) {
            matrix_element += ints->tei_ab(p, i, p, a);
        }
        if (Ib.get_bit(p)) {
            matrix_element += ints->tei_bb(p, i, p, a);
        }
    }
    return (Ib.slater_sign(i, a) * matrix_element);
}

double slater_rules_double_alpha_alpha(String Ia, String Ja,
                                       const std::shared_ptr<ActiveSpaceIntegrals>& ints) {

    String IJb = Ia ^ Ja;

    String Ia_sub = Ia & IJb;
    uint64_t i = Ia_sub.find_and_clear_first_one();
    uint64_t j = Ia_sub.find_first_one();

    String Ja_sub = Ja & IJb;
    uint64_t k = Ja_sub.find_and_clear_first_one();
    uint64_t l = Ja_sub.find_first_one();

    return Ia.slater_sign(i, j) * Ja.slater_sign(k, l) * ints->tei_aa(i, j, k, l);
}

double slater_rules_double_beta_beta(String Ib, String Jb,
                                     const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    String IJb = Ib ^ Jb;

    String Ib_sub = Ib & IJb;
    uint64_t i = Ib_sub.find_and_clear_first_one();
    uint64_t j = Ib_sub.find_first_one();

    String Jb_sub = Jb & IJb;
    uint64_t k = Jb_sub.find_and_clear_first_one();
    uint64_t l = Jb_sub.find_first_one();

    return Ib.slater_sign(i, j) * Jb.slater_sign(k, l) * ints->tei_bb(i, j, k, l);
}

double slater_rules_double_alpha_beta_pre(int i, int a, String Ib, String Jb,
                                          const std::shared_ptr<ActiveSpaceIntegrals>& ints) {
    String Ib_xor_Jb = Ib ^ Jb;
    uint64_t j = Ib_xor_Jb.find_and_clear_first_one();
    uint64_t b = Ib_xor_Jb.find_first_one();
    return Ib.slater_sign(j, b) * ints->tei_ab(i, j, a, b);
}

Determinant common_occupation(const Determinant& lhs, const Determinant& rhs) { return lhs & rhs; }

Determinant different_occupation(const Determinant& lhs, const Determinant& rhs) {
    return lhs ^ rhs;
}

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
Determinant union_occupation(const Determinant& lhs, const Determinant& rhs) { return lhs | rhs; }

void enforce_spin_completeness(std::vector<Determinant>& det_space, int nmo) {
    std::unordered_map<Determinant, bool, Determinant::Hash> det_map;
    // Add all determinants to the map, assume set is mostly spin complete
    for (auto& I : det_space) {
        det_map[I] = true;
    }
    // Loop over determinants
    size_t ndet_added = 0;
    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    for (size_t I = 0, det_size = det_space.size(); I < det_size; ++I) {
        const Determinant& det = det_space[I];
        // outfile->Printf("\n  Original determinant: %s", det.str().c_str());
        for (int i = 0; i < nmo; ++i) {
            closed[i] = open[i] = 0;
            open_bits[i] = false;
        }
        int naopen = 0;
        int nbopen = 0;
        int nclosed = 0;
        for (int i = 0; i < nmo; ++i) {
            if (det.get_alfa_bit(i) and (not det.get_beta_bit(i))) {
                open[naopen + nbopen] = i;
                naopen += 1;
            } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)) {
                open[naopen + nbopen] = i;
                nbopen += 1;
            } else if (det.get_alfa_bit(i) and det.get_beta_bit(i)) {
                closed[nclosed] = i;
                nclosed += 1;
            }
        }

        if (naopen + nbopen == 0)
            continue;

        // Generate the strings 1111100000
        //                      {nao}{nbo}
        for (int i = 0; i < nbopen; ++i)
            open_bits[i] = false; // 0
        for (int i = nbopen; i < naopen + nbopen; ++i)
            open_bits[i] = true; // 1
        do {
            Determinant new_det;
            for (int c = 0; c < nclosed; ++c) {
                new_det.set_alfa_bit(closed[c], true);
                new_det.set_beta_bit(closed[c], true);
            }
            for (int o = 0; o < naopen + nbopen; ++o) {
                if (open_bits[o]) { //? not
                    new_det.set_alfa_bit(open[o], true);
                } else {
                    new_det.set_beta_bit(open[o], true);
                }
            }
            if (det_map.count(new_det) == 0) {
                det_space.push_back(new_det);
                det_map[new_det] = true;
                // outfile->Printf("\n  added determinant:    %s", new_det.str().c_str());
                ndet_added++;
            }
        } while (std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
    }
}

} // namespace forte
