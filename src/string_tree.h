/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _string_tree_h_
#define _string_tree_h_

#include <bitset>
#include <unordered_map>

namespace psi {
namespace forte {

using sbit_t = unsigned long int;

class string_tree
{
public:
    // Build the string tree from a list of determinants
    string_tree(size_t ndets, std::vector<Determinants>);
    size_t ndets_;
    std::vector<sbit_t> sorted_a_;
    std::vector<sbit_t> sorted_b_;
    std::vector<sbit_t> sorted_ab_;
    std::vector<sbit_t> sorted_ba_;
    std::vector<double> C_ab_;
    std::vector<double> C_ba_;
};

}
} // End Namespaces

#endif // _string_tree_h_
