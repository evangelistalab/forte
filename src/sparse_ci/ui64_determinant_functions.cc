/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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
#include <unordered_map>

#include "psi4/libmints/dimension.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include "../fci/fci_integrals.h"

#include "ui64_determinant.h"
#include "stl_bitset_determinant.h"

namespace psi {
namespace forte {

double slater_rules_single_alpha(uint64_t Ib, uint64_t Ia, uint64_t Ja,
                                 const std::shared_ptr<FCIIntegrals>& ints) {
    uint64_t IJa = Ia ^ Ja;
    uint64_t i = lowest_one_idx(IJa);
    IJa = clear_lowest_one(IJa);
    uint64_t a = lowest_one_idx(IJa);

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    // find common bits
    IJa = Ia & Ja;
    for (int p = 0; p < 64; ++p) {
        if (ui64_get_bit(Ia, p)) {
            matrix_element += ints->tei_aa(i, p, a, p);
        }
        if (ui64_get_bit(Ib, p)) {
            matrix_element += ints->tei_ab(i, p, a, p);
        }
    }
    return (ui64_slater_sign(Ia, i, a) * matrix_element);
}

double slater_rules_single_beta(uint64_t Ia, uint64_t Ib, uint64_t Jb,
                                const std::shared_ptr<FCIIntegrals>& ints) {
    uint64_t IJb = Ib ^ Jb;
    uint64_t i = lowest_one_idx(IJb);
    IJb = clear_lowest_one(IJb);
    uint64_t a = lowest_one_idx(IJb);

    // Diagonal contribution
    double matrix_element = ints->oei_b(i, a);
    // find common bits
    IJb = Ib & Jb;
    for (int p = 0; p < 64; ++p) {
        if (ui64_get_bit(Ia, p)) {
            matrix_element += ints->tei_ab(p, i, p, a);
        }
        if (ui64_get_bit(Ib, p)) {
            matrix_element += ints->tei_bb(p, i, p, a);
        }
    }
    return (ui64_slater_sign(Ib, i, a) * matrix_element);
}

double slater_rules_double_alpha_alpha(uint64_t Ia, uint64_t Ja,
                                       const std::shared_ptr<FCIIntegrals>& ints) {
    uint64_t IJb = Ia ^ Ja;

    uint64_t Ia_sub = Ia & IJb;
    uint64_t i = lowest_one_idx(Ia_sub);
    Ia_sub = clear_lowest_one(Ia_sub);
    uint64_t j = lowest_one_idx(Ia_sub);

    uint64_t Ja_sub = Ja & IJb;
    uint64_t k = lowest_one_idx(Ja_sub);
    Ja_sub = clear_lowest_one(Ja_sub);
    uint64_t l = lowest_one_idx(Ja_sub);

    return ui64_slater_sign(Ia, i, j) * ui64_slater_sign(Ja, k, l) * ints->tei_aa(i, j, k, l);
}

double slater_rules_double_beta_beta(uint64_t Ib, uint64_t Jb,
                                     const std::shared_ptr<FCIIntegrals>& ints) {
    uint64_t IJb = Ib ^ Jb;

    uint64_t Ib_sub = Ib & IJb;
    uint64_t i = lowest_one_idx(Ib_sub);
    Ib_sub = clear_lowest_one(Ib_sub);
    uint64_t j = lowest_one_idx(Ib_sub);

    uint64_t Jb_sub = Jb & IJb;
    uint64_t k = lowest_one_idx(Jb_sub);
    Jb_sub = clear_lowest_one(Jb_sub);
    uint64_t l = lowest_one_idx(Jb_sub);

    return ui64_slater_sign(Ib, i, j) * ui64_slater_sign(Jb, k, l) * ints->tei_bb(i, j, k, l);
}

double slater_rules_double_alpha_beta_pre(int i, int a, uint64_t Ib, uint64_t Jb,
                                          const std::shared_ptr<FCIIntegrals>& ints) {
//    outfile->Printf("\n %zu %zu", Ib, Jb);
    uint64_t Ib_xor_Jb = Ib ^ Jb;
    uint64_t j = lowest_one_idx(Ib_xor_Jb);
    Ib_xor_Jb = clear_lowest_one(Ib_xor_Jb);
    uint64_t b = lowest_one_idx(Ib_xor_Jb);
//    outfile->Printf("\n  i = %d, j = %d, a = %d, b = %d", i, j, a, b);
    return ui64_slater_sign(Ib, j, b) * ints->tei_ab(i, j, a, b);
}

UI64Determinant common_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs) {
    UI64Determinant result;
    result.set_alfa_bits(lhs.get_alfa_bits() & rhs.get_alfa_bits());
    result.set_beta_bits(lhs.get_beta_bits() & rhs.get_beta_bits());
    return result;
}

UI64Determinant different_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs) {
    UI64Determinant result;
    result.set_alfa_bits(lhs.get_alfa_bits() ^ rhs.get_alfa_bits());
    result.set_beta_bits(lhs.get_beta_bits() ^ rhs.get_beta_bits());
    return result;
}

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
UI64Determinant union_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs) {
    UI64Determinant result;
    result.set_alfa_bits(lhs.get_alfa_bits() | rhs.get_alfa_bits());
    result.set_beta_bits(lhs.get_beta_bits() | rhs.get_beta_bits());
    return result;
}

void enforce_spin_completeness(std::vector<UI64Determinant>& det_space, int nmo) {
    std::unordered_map<UI64Determinant, bool, UI64Determinant::Hash> det_map;

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
        const UI64Determinant& det = det_space[I];
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
            UI64Determinant new_det;
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
    // if( ndet_added > 0 ){
    //    outfile->Printf("\n\n  Determinant space is spin incomplete!");
    //    outfile->Printf("\n  %zu more determinants were needed.", ndet_added);
    //}else{
    //    outfile->Printf("\n\n  Determinant space is spin complete.");
    //}
}

template <>
UI64Determinant make_det<UI64Determinant, STLBitsetDeterminant>(const STLBitsetDeterminant& d) {
    UI64Determinant ui64_d;
    for (int i = 0; i < 64; ++i) {
        ui64_d.set_alfa_bit(i, d.get_alfa_bit(i));
        ui64_d.set_beta_bit(i, d.get_beta_bit(i));
    }
    return ui64_d;
}

template <> UI64Determinant make_det<UI64Determinant, UI64Determinant>(const UI64Determinant& d) {
    return d;
}
}
}
