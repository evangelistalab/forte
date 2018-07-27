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

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif


#include "stl_bitset_determinant.h"
#include "ui64_determinant.h"

namespace psi {
namespace forte {

#define USE_builtin_popcountll 1

/**
 * @brief compute the parity (+/-) of an unsigned 64 bit integer
 * @param number x
 * @return the parity of x
 */
double parity(uint64_t x) { return (x % 2 == 0) ? 1.0 : -1.0; }

/**
 * @brief return the value of bit in a uint64_t
 * @param x the integer to test
 * @param n the position of the bit
 * @return the value of the bit
 */
bool ui64_get_bit(uint64_t x, uint64_t n) { return (0 != (x & (uint64_t(1) << n))); }

/**
 * @brief count the number of bit set to 1 in a uint64_t
 * @param x the integer to test
 * @return the number of bit that are set
 */
uint64_t ui64_bit_count(uint64_t x) {
#ifdef __SSE4_2__
    return _mm_popcnt_u64(x);
//    return __builtin_popcountll(x);
#else
//#ifdef SSE42_FLAG
//#ifdef USE_builtin_popcountll
    // optimized version using popcnt
//    return __builtin_popcountll(x);
//#else
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

double ui64_slater_sign(uint64_t x, int n) {
    // This implementation is 20 x times faster than one based on for loops
    // First build a mask with n bit set. Then & with string.
    // Example for 16 bit string
    //                 n            (n = 5)
    // want       11111000 00000000
    // mask       11111111 11111111
    // mask << 11 00000000 00011111 11 = 16 - 5
    // mask >> 11 11111000 00000000
    //
    // Note: This strategy does not work when n = 0 because ~0 << 64 = ~0 (!!!)
    // so we treat this case separately
    if (n == 0)
        return 1.0;
    uint64_t mask = ~0;
    mask = mask << (64 - n);             // make a string with n bits set
    mask = mask >> (64 - n);             // move it right by n
    mask = x & mask;                     // intersect with string
    mask = ui64_bit_count(mask);         // count bits in between
    return (mask % 2 == 0) ? 1.0 : -1.0; // compute sign
}

double ui64_slater_sign(uint64_t x, int m, int n) {
    // This implementation is a bit faster than one based on for loops
    // First build a mask with bit set between positions m and n. Then & with string.
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

UI64Determinant::UI64Determinant() : a_(0), b_(0) {}

UI64Determinant::UI64Determinant(const std::vector<bool>& occupation) : a_(0), b_(0) {
    int size = occupation.size() / 2;
    for (int p = 0; p < size; ++p)
        set_alfa_bit(p, occupation[p]);
    for (int p = 0; p < size; ++p)
        set_beta_bit(p, occupation[size + p]);
}

UI64Determinant::UI64Determinant(const std::vector<bool>& occupation_a,
                                 const std::vector<bool>& occupation_b)
    : a_(0), b_(0) {
    int size = occupation_a.size();
    for (int p = 0; p < size; ++p) {
        set_alfa_bit(p, occupation_a[p]);
        set_beta_bit(p, occupation_b[p]);
    }
}

UI64Determinant::bit_t UI64Determinant::get_alfa_bits() const { return a_; }

UI64Determinant::bit_t UI64Determinant::get_beta_bits() const { return b_; }

void UI64Determinant::set_alfa_bits(bit_t x) { a_ = x; }

void UI64Determinant::set_beta_bits(bit_t x) { b_ = x; }

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

int UI64Determinant::count_alfa() const { return ui64_bit_count(a_); }

int UI64Determinant::count_beta() const { return ui64_bit_count(b_); }

int UI64Determinant::npair() const { return ui64_bit_count(a_ & b_); }

std::vector<int> UI64Determinant::get_alfa_occ(int norb) const {
    std::vector<int> occ;
    for (int p = 0; p < norb; ++p) {
        if (get_alfa_bit(p)) {
            occ.push_back(p);
        }
    }
    return occ;
}

std::vector<int> UI64Determinant::get_beta_occ(int norb) const {
    std::vector<int> occ;
    for (int p = 0; p < norb; ++p) {
        if (get_beta_bit(p)) {
            occ.push_back(p);
        }
    }
    return occ;
}

std::vector<int> UI64Determinant::get_alfa_vir(int norb) const {
    std::vector<int> vir;
    for (int p = 0; p < norb; ++p) {
        if (not get_alfa_bit(p)) {
            vir.push_back(p);
        }
    }
    return vir;
}

std::vector<int> UI64Determinant::get_beta_vir(int norb) const {
    std::vector<int> vir;
    for (int p = 0; p < norb; ++p) {
        if (not get_beta_bit(p)) {
            vir.push_back(p);
        }
    }
    return vir;
}

double UI64Determinant::create_alfa_bit(int n) {
    if (get_alfa_bit(n))
        return 0.0;
    set_alfa_bit(n, true);
    return slater_sign_a(n);
}

double UI64Determinant::create_beta_bit(int n) {
    if (get_beta_bit(n))
        return 0.0;
    set_beta_bit(n, true);
    return slater_sign_b(n);
}

double UI64Determinant::destroy_alfa_bit(int n) {
    if (not get_alfa_bit(n))
        return 0.0;
    set_alfa_bit(n, false);
    return slater_sign_a(n);
}

double UI64Determinant::destroy_beta_bit(int n) {
    if (not get_beta_bit(n))
        return 0.0;
    set_beta_bit(n, false);
    return slater_sign_b(n);
}

std::vector<std::vector<int>> UI64Determinant::get_asym_occ(int norb,
                                                            std::vector<int> act_mo) const {

    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (get_alfa_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

std::vector<std::vector<int>> UI64Determinant::get_bsym_occ(int norb,
                                                            std::vector<int> act_mo) const {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (get_beta_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

std::vector<std::vector<int>> UI64Determinant::get_asym_vir(int norb,
                                                            std::vector<int> act_mo) const {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not get_alfa_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

std::vector<std::vector<int>> UI64Determinant::get_bsym_vir(int norb,
                                                            std::vector<int> act_mo) const {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not get_beta_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}
void UI64Determinant::zero_spin(DetSpinType spin_type) {
    if (spin_type == DetSpinType::Alpha) {
        a_ = bit_t(0);
    } else {
        b_ = bit_t(0);
    }
}

std::string UI64Determinant::str(int n) const {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (get_alfa_bit(p) and get_beta_bit(p)) {
            s += "2";
        } else if (get_alfa_bit(p) and not get_beta_bit(p)) {
            s += "+";
        } else if (not get_alfa_bit(p) and get_beta_bit(p)) {
            s += "-";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

double UI64Determinant::slater_sign_a(int n) const { return ui64_slater_sign(a_, n); }

double UI64Determinant::slater_sign_aa(int n, int m) const { return ui64_slater_sign(a_, n, m); }

double UI64Determinant::slater_sign_b(int n) const {
    return ui64_slater_sign(b_, n) * (ui64_bit_count(a_) % 2 == 0 ? 1.0 : -1.0);
}

double UI64Determinant::slater_sign_bb(int n, int m) const { return ui64_slater_sign(b_, n, m); }

double UI64Determinant::slater_sign_aaaa(int i, int j, int a, int b) const {
    if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
        (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_aa(i, j) * slater_sign_aa(a, b));
        } else {
            return (slater_sign_aa(i, j) * slater_sign_aa(a, b));
        }
    } else {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_aa(i, b) * slater_sign_aa(j, a));
        } else {
            return (slater_sign_aa(i, a) * slater_sign_aa(j, b));
        }
    }
}

double UI64Determinant::slater_sign_bbbb(int i, int j, int a, int b) const {
    if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
        (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_bb(i, j) * slater_sign_bb(a, b));
        } else {
            return (slater_sign_bb(i, j) * slater_sign_bb(a, b));
        }
    } else {
        if ((i < j) ^ (a < b)) {
            return (-1.0 * slater_sign_bb(i, b) * slater_sign_bb(j, a));
        } else {
            return (slater_sign_bb(i, a) * slater_sign_bb(j, b));
        }
    }
}

double UI64Determinant::single_excitation_a(int i, int a) {
    set_alfa_bit(i, false);
    set_alfa_bit(a, true);
    return slater_sign_aa(i, a);
}

double UI64Determinant::single_excitation_b(int i, int a) {
    set_beta_bit(i, false);
    set_beta_bit(a, true);

    return slater_sign_bb(i, a);
}

double UI64Determinant::double_excitation_aa(int i, int j, int a, int b) {
    set_alfa_bit(i, false);
    set_alfa_bit(j, false);
    set_alfa_bit(b, true);
    set_alfa_bit(a, true);
    return slater_sign_aaaa(i, j, a, b);
}

double UI64Determinant::double_excitation_ab(int i, int j, int a, int b) {
    set_alfa_bit(i, false);
    set_beta_bit(j, false);
    set_beta_bit(b, true);
    set_alfa_bit(a, true);
    return slater_sign_aa(i, a) * slater_sign_bb(j, b);
}

double UI64Determinant::double_excitation_bb(int i, int j, int a, int b) {
    set_beta_bit(i, false);
    set_beta_bit(j, false);
    set_beta_bit(b, true);
    set_beta_bit(a, true);
    return slater_sign_bbbb(i, j, a, b);
}

UI64Determinant& UI64Determinant::flip() {
    a_ = ~a_;
    b_ = ~b_;
    return *this;
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

double spin2(const UI64Determinant& lhs, const UI64Determinant& rhs) {
    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = ui64_bit_count(lhs.get_alfa_bits() ^ rhs.get_alfa_bits()) / 2;
    int nbdiff = ui64_bit_count(lhs.get_beta_bits() ^ rhs.get_beta_bits()) / 2;
    int na = ui64_bit_count(lhs.get_alfa_bits());
    int nb = ui64_bit_count(lhs.get_beta_bits());
    int npair = lhs.npair();

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta - Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here is a bit complex
        for (int p = 0; p < 64; ++p) {
            if (rhs.get_alfa_bit(p) and lhs.get_beta_bit(p) and (not rhs.get_beta_bit(p)) and
                (not lhs.get_alfa_bit(p)))
                i = p;
            if (rhs.get_beta_bit(p) and lhs.get_alfa_bit(p) and (not rhs.get_alfa_bit(p)) and
                (not lhs.get_beta_bit(p)))
                j = p;
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign = rhs.slater_sign_a(i) * rhs.slater_sign_b(j) * lhs.slater_sign_a(j) *
                          lhs.slater_sign_b(i);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}
} // namespace forte
} // namespace psi
