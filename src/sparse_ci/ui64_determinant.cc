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

namespace psi {
namespace forte {

bool ui64_get_bit(uint64_t x, uint64_t n) { return (0 != (x & (uint64_t(1) << n))); }

uint64_t ui64_bit_count(uint64_t x) {
    x = (0x5555555555555555UL & x) + (0x5555555555555555UL & (x >> 1));
    x = (0x3333333333333333UL & x) + (0x3333333333333333UL & (x >> 2));
    x = (0x0f0f0f0f0f0f0f0fUL & x) + (0x0f0f0f0f0f0f0f0fUL & (x >> 4));
    x = (0x00ff00ff00ff00ffUL & x) + (0x00ff00ff00ff00ffUL & (x >> 8));
    x = (0x0000ffff0000ffffUL & x) + (0x0000ffff0000ffffUL & (x >> 16));
    x = (0x00000000ffffffffUL & x) + (0x00000000ffffffffUL & (x >> 32));
    return x;
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

uint64_t lowest_one_idx(uint64_t x) {
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
}

uint64_t clear_lowest_one(uint64_t x)
// Return word where the lowest bit set in x is cleared
// Return 0 for input == 0
{
    return x & (x - 1);
}

// inline std::vector<int> get_set(uint64_t x, uint64_t range) {
//    uint64_t mask = (bit_t(1) << range) - bit_t(1);
//    x = x & mask;
//    std::vector<int> r(bit_count(x));
//    bit_t index = lowest_one_idx(x);
//    int i = 0;
//    while (index != -1) {
//        r[i] = index;
//        x = clear_lowest_one(x);
//        index = lowest_one_idx(x);
//        i++;
//    }
//    return r;
//}

double ui64_slater_sign(uint64_t x, int m, int n) {
    double sign = 1.0;
    for (int i = m + 1; i < n; ++i) { // This runs up to the operator before n
        if (ui64_get_bit(x, i))
            sign *= -1.0;
    }
    for (int i = n + 1; i < m; ++i) {
        if (ui64_get_bit(x, i)) {
            sign *= -1.0;
        }
    }
    return (sign);
}

std::tuple<double, size_t, size_t> ui64_slater_sign_single(uint64_t l, uint64_t r) {
    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    size_t j, b;
    for (int p = 0; p < 64; ++p) {
        const bool lb_p = ui64_get_bit(l, p);
        const bool rb_p = ui64_get_bit(r, p);
        if (lb_p ^ rb_p) {
            j = lb_p ? p : j;
            b = rb_p ? p : b;
        }
    }
    return std::make_tuple(ui64_slater_sign(l, j, b), j, b);
}
// lhs.slater_sign_bb(j, l) * tei_ab_[i * nmo3_ + j * nmo2_ + k * nmo_ + l]
// double ui64_slater_sign_single(uint64_t x) {
//    double sign = 1.0;
//    //  uint64_t m = lowest_one_idx(x);
//    //  x = clear_lowest_one(x);
//    //  uint64_t n = lowest_one_idx(x);

//    bool count = false;
//    for (int i = 0; i < 64; ++i) { // This runs up to the operator before n
//        if (not count) {
//            if (ui64_get_bit(x, i)) {
//                count = true;
//            }
//        } else {
//            sign *= -1.0;
//        }
//    }

//    //  for (int i = m + 1; i < n; ++i) { // This runs up to the operator before n
//    //      if (ui64_get_bit(x, i))
//    //          sign *= -1.0;
//    //  }
//    //  for (int i = n + 1; i < m; ++i) {
//    //      if (ui64_get_bit(x, i)) {
//    //          sign *= -1.0;
//    //      }
//    //  }
//    return (sign);
//}

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

UI64Determinant::bit_t UI64Determinant::get_bits(STLBitsetDeterminant::SpinType spin_type) const {
    return (spin_type == STLBitsetDeterminant::SpinType::AlphaSpin) ? a_ : b_;
}

void UI64Determinant::zero_spin(STLBitsetDeterminant::SpinType spin_type) {
    if (spin_type == STLBitsetDeterminant::SpinType::AlphaSpin) {
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
