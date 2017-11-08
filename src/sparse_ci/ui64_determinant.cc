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

#include "ui64_determinant.h"
#include "nmmintrin.h"

namespace psi {
namespace forte {

#define USE_builtin_popcountll 1

bool ui64_get_bit(uint64_t x, uint64_t n) { return (0 != (x & (uint64_t(1) << n))); }

uint64_t ui64_bit_count(uint64_t x) {
    return _mm_popcnt_u64(x);
#ifdef USE_builtin_popcountll
// optimized version using popcnt
// return __builtin_popcountll(x);
#else
    // version based on bitwise operations
    x = (0x5555555555555555UL & x) + (0x5555555555555555UL & (x >> 1));
    x = (0x3333333333333333UL & x) + (0x3333333333333333UL & (x >> 2));
    x = (0x0f0f0f0f0f0f0f0fUL & x) + (0x0f0f0f0f0f0f0f0fUL & (x >> 4));
    x = (0x00ff00ff00ff00ffUL & x) + (0x00ff00ff00ff00ffUL & (x >> 8));
    x = (0x0000ffff0000ffffUL & x) + (0x0000ffff0000ffffUL & (x >> 16));
    x = (0x00000000ffffffffUL & x) + (0x00000000ffffffffUL & (x >> 32));
    return x;
#endif
    //    x = ((x>>1) & 0x5555555555555555UL) + (x & 0x5555555555555555UL);
    //    x = ((x>>2) & 0x3333333333333333UL) + (x & 0x3333333333333333UL);
    //    x = ((x>>4) + x) & 0x0f0f0f0f0f0f0f0fUL;
    //    x+=x>>8;
    //    x += x>>16;
    //    x += x>>32;
    //    return x & 0xff;

    //    x -= (x>>1) & 0x5555555555555555UL;
    //    x = ((x>>2) & 0x3333333333333333UL) + (x & 0x3333333333333333UL); // 0-4 in 4 bits
    //    x = ((x>>4) + x) & 0x0f0f0f0f0f0f0f0fUL; // 0-8 in 8 bits
    //    x *= 0x0101010101010101UL;
    //    return x>>56;
}

/// Returns the index of the least significant 1-bit of x, or if x is zero, returns ~0.
uint64_t lowest_one_idx(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    // optimized version using builtin functions
    return __builtin_ffsll(x) - 1;
#else
    // version based on bitwise operations
    if (1 >= x)
        return x - 1; // 0 if 1, ~0 if 0
    uint64_t r = 0;
    x &= -x; // isolate lowest bit
    if (x & 0xffffffff00000000UL)
        r += 32;
    if (x & 0xffff0000ffff0000UL)
        r += 16;
    if (x & 0xff00ff00ff00ff00UL)
        r += 8;
    if (x & 0xf0f0f0f0f0f0f0f0UL)
        r += 4;
    if (x & 0xccccccccccccccccUL)
        r += 2;
    if (x & 0xaaaaaaaaaaaaaaaaUL)
        r += 1;
    return r;
#endif
}

uint64_t clear_lowest_one(uint64_t x)
// Return word where the lowest bit set in x is cleared
// Return 0 for input == 0
{
    return x & (x - 1);
}

double ui64_slater_sign(uint64_t x, int m, int n) {
    // Example for 16 bit string
    //              m  n            (m = 2, n = 5)
    // want       00011000 00000000
    // mask       11111111 11111111
    // mask << 14 00000000 00000011 14 = 16 + 1 - |5-2|
    // mask >> 11 00011000 00000000 11 = 16 - |5-2| -  min(2,5)
    // the mask should have |m - n| - 1 bits turned on
    uint64_t gap = std::abs(m - n);
    if (gap < 2) { // special cases
        return 1.0;
    }
    uint64_t mask = ~0;
    mask = mask << (65 - gap);                  // make a string with |m - n| - 1 bits set
    mask = mask >> (64 - gap - std::min(m, n)); // move it right after min(m, n)
    mask = x & mask;                            // intersect with string
    mask = ui64_bit_count(mask);                // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0;        // compute sign
}

std::tuple<double, size_t, size_t> ui64_slater_sign_single(uint64_t l, uint64_t r) {
    uint64_t lr = l ^ r;
    uint64_t j = lowest_one_idx(lr);
    lr = clear_lowest_one(lr);
    uint64_t b = lowest_one_idx(lr);
    return std::make_tuple(ui64_slater_sign(l, j, b), j, b);
}

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
    uint64_t Ib_xor_Jb = Ib ^ Jb;
    uint64_t j = lowest_one_idx(Ib_xor_Jb);
    Ib_xor_Jb = clear_lowest_one(Ib_xor_Jb);
    uint64_t b = lowest_one_idx(Ib_xor_Jb);
    return ui64_slater_sign(Ib, j, b) * ints->tei_ab(i, j, a, b);
}

UI64Determinant::UI64Determinant() {}
UI64Determinant::UI64Determinant(const STLBitsetDeterminant& d) {
    for (int i = 0; i < 64; ++i) {
        set_alfa_bit(i, d.get_alfa_bit(i));
        set_beta_bit(i, d.get_beta_bit(i));
    }
}

UI64Determinant::bit_t UI64Determinant::get_alfa_bits() const { return a_; }

UI64Determinant::bit_t UI64Determinant::get_beta_bits() const { return b_; }

bool UI64Determinant::get_alfa_bit(UI64Determinant::bit_t n) const {
    return (0 != (a_ & (bit_t(1) << n)));
}

bool UI64Determinant::get_beta_bit(UI64Determinant::bit_t n) const {
    return (0 != (b_ & (bit_t(1) << n)));
}

/// Set the value of an alpha bit
void UI64Determinant::set_alfa_bit(UI64Determinant::bit_t n, bool v) {
    if (v) {
        a_ |= (bit_t(1) << n);
    } else {
        a_ &= ~(bit_t(1) << n);
    }
}
//            alfa_bits_ ^= (-bit_t(v) ^ alfa_bits_) & (1 << n);}
/// Set the value of a beta bit
void UI64Determinant::set_beta_bit(UI64Determinant::bit_t n, bool v) {
    if (v) {
        b_ |= (bit_t(1) << n);
    } else {
        b_ &= ~(bit_t(1) << n);
    }
}

UI64Determinant::bit_t UI64Determinant::get_bits(DetSpinType spin_type) const {
    return (spin_type == DetSpinType::Alpha) ? a_ : b_;
}

void UI64Determinant::zero_spin(DetSpinType spin_type) {
    if (spin_type == DetSpinType::Alpha) {
        a_ = bit_t(0);
    } else {
        b_ = bit_t(0);
    }
}

bool UI64Determinant::less_than(const UI64Determinant& rhs, const UI64Determinant& lhs) {
    if (rhs.b_ < lhs.b_) {
        return true;
    } else if (rhs.b_ > lhs.b_) {
        return false;
    }
    return rhs.a_ < lhs.a_;
}

bool UI64Determinant::reverse_less_than(const UI64Determinant& rhs, const UI64Determinant& lhs) {
    if (rhs.a_ < lhs.a_) {
        return true;
    } else if (rhs.a_ > lhs.a_) {
        return false;
    }
    return rhs.b_ < lhs.b_;
}

bool UI64Determinant::operator==(const UI64Determinant& lhs) const {
    return ((a_ == lhs.a_) and (b_ == lhs.b_));
}

bool UI64Determinant::operator<(const UI64Determinant& lhs) const {
    if (b_ < lhs.b_) {
        return true;
    } else if (b_ > lhs.b_) {
        return false;
    }
    return a_ < lhs.a_;
}
}
}
