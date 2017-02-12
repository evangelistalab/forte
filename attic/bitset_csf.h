/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _bitset_csf_h_
#define _bitset_csf_h_

#include "boost/dynamic_bitset.hpp"

#include "integrals.h"

namespace psi {
namespace forte {

/**
 * A class to store a configuration state function (CSF) using Boost's
 * dynamic_bitset.
 *
 * The CSF is represented by a pair of strings that specify the occupation
 * of the doubly and singly occupied molecular orbitals.
 * (excluding frozen core and virtual orbitals).
 *
 * |CSF> = |doubly occupied> x |singly occupied>
 *
 * The strings are represented using an array of bits, and the
 * following convention is used here:
 * true <-> 1 or 2
 * false <-> 0
 */
class BitsetCSF {
  public:
    /// Define the bit type (bit_t)
    using bit_t = boost::dynamic_bitset<>;

    // ==> Class Constructor and Destructor <==

    /// Construct an empty CSF
    BitsetCSF();

    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit BitsetCSF(const std::vector<bool>& occupation_a,
                       const std::vector<bool>& occupation_b, short s, short ms,
                       short index = 0);

    void print() const;
    std::string str() const;

  private:
    // ==> Private Class Data <==

    /// The occupation vector for the doubly occupied MOs (does not include the
    /// frozen orbitals)
    bit_t docc_;
    /// The occupation vector for the singly occupied MOs (does not include the
    /// frozen orbitals)
    bit_t socc_;
    /// The S value
    short s_;
    /// The M_S value
    short ms_;
    /// The index of this CSF
    short index_;
};
}
} // End Namespaces

#endif // _bitset_csf_h_
