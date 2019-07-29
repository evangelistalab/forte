/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#if 0

#ifndef _bitset_string_h_
#define _bitset_string_h_

#include <unordered_map>
#include <bitset>

#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"


namespace forte {

/**
 * A class to store a Slater determinant using the STL bitset container.
 *
 * The determinant is represented by a pair of alpha/beta strings
 * that specify the occupation of each molecular orbital
 * (excluding frozen core and virtual orbitals).
 *
 * |Det> = |I>
 *
 * The strings are represented using one array of bits of size 2 x nmo,
 * and the following convention is used here:
 * true <-> 1
 * false <-> 0
 */
class STLBitsetString {
  public:
    using bit_t = std::bitset<Determinant::num_str_bits>;

    // Class Constructor and Destructor
    /// Construct an empty occupation string
    STLBitsetString();

    /// Construct String from an occupation vector, spin unspecified
    explicit STLBitsetString(const std::vector<int>& occupation);
    explicit STLBitsetString(const std::vector<bool>& occupation);
    /// Construnct a determinant from a bitset object
    explicit STLBitsetString(const bit_t& bits);

    /// Equal operator
    bool operator==(const STLBitsetString& lhs) const;
    /// Less than operator
    bool operator<(const STLBitsetString& lhs) const;
    /// XOR operator
    STLBitsetString operator^(const STLBitsetString& lhs) const;

    /// Set the dimension
    void set_nmo(int nmo);

    /// Get a pointer to the alpha bits
    const bit_t& bits() const;

    /// Return the value of a bit
    bool get_bit(int n) const;

    /// Set the value of a bit
    void set_bit(int n, bool value);

    /// Get a vector of the  bits
    std::vector<bool> get_bits_vector_bool();

    /// Return a vector of occupied orbitals
    std::vector<int> get_occ();
    /// Return a vector of virtual orbitals
    std::vector<int> get_vir();

    /// Print the occupation string
    void print() const;
    /// Save the occupation string  as a std::string
    std::string str() const;
    /// Return the sign of a_n applied to string I
    double SlaterSign(int n);

  public:
    // Object Data
    /// The occupation vector (does not include the frozen orbitals)
    bit_t bits_;

    // Static data
    /// Number of non-frozen molecular orbitals
    static int nmo_;

    /// Number of non-zero bits
    double get_nocc();

    struct Hash {
        std::size_t operator()(const forte::STLBitsetString& bs) const {
            return std::hash<bit_t>()(bs.bits_);
        }
    };
};

using string_vec = std::vector<STLBitsetString>;
template <typename T = double>
using string_hash = std::unordered_map<STLBitsetString, T, STLBitsetString::Hash>;
using string_hash_it = std::unordered_map<STLBitsetString, double, STLBitsetString::Hash>::iterator;
}

#endif // _bitset_string_h_

#endif
