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

#ifndef _ui64_determinant_h_
#define _ui64_determinant_h_

#include "stl_bitset_determinant.h"
#include "../fci/fci_integrals.h"

namespace psi {
namespace forte {

class UI64Determinant {
  public:
    using bit_t = uint64_t;
    UI64Determinant();
    UI64Determinant(const STLBitsetDeterminant& d);

    /// Equal operator
    bool operator==(const UI64Determinant& lhs) const;
    /// Less than operator
    bool operator<(const UI64Determinant& lhs) const;

    /// Less than operator
    static bool less_than(const UI64Determinant& rhs, const UI64Determinant& lhs);
    /// Reverse string ordering
    static bool reverse_less_than(const UI64Determinant& i, const UI64Determinant& j);

    bit_t get_bits(DetSpinType spin_type) const;
    /// Return the value of an alpha bit
    bit_t get_alfa_bits() const;
    /// Return the value of a beta bit
    bit_t get_beta_bits() const;
    /// Return the value of an alpha bit
    bool get_alfa_bit(bit_t n) const;
    /// Return the value of a beta bit
    bool get_beta_bit(bit_t n) const;
    void set_alfa_bit(bit_t n, bool v);
    void set_beta_bit(bit_t n, bool v);
    void zero_spin(DetSpinType spin_type);

  private:
    bit_t a_;
    bit_t b_;
};

// Return number of bits set
uint64_t ui64_bit_count(uint64_t x);
bool ui64_get_bit(uint64_t x, uint64_t n);
uint64_t lowest_one_idx(uint64_t x);
uint64_t clear_lowest_one(uint64_t x);

double ui64_slater_sign(uint64_t x, int m, int n);
std::tuple<double, size_t, size_t> ui64_slater_sign_single(uint64_t l, uint64_t r);

double slater_rules_single_alpha(uint64_t Ib, uint64_t Ia, uint64_t Ja,
                                 const std::shared_ptr<FCIIntegrals>& ints);
double slater_rules_double_alpha_alpha(uint64_t Ia, uint64_t Ja,
                                       const std::shared_ptr<FCIIntegrals>& ints);
double slater_rules_single_beta(uint64_t Ia, uint64_t Ib, uint64_t Jb,
                                const std::shared_ptr<FCIIntegrals>& ints);
double slater_rules_double_beta_beta(uint64_t Ib, uint64_t Jb,
                                     const std::shared_ptr<FCIIntegrals>& ints);
double slater_rules_double_alpha_beta_pre(int i, int a, uint64_t Ib, uint64_t Jb,
                                          const std::shared_ptr<FCIIntegrals>& ints);
}
}

#endif // _ui64_determinant_h_
