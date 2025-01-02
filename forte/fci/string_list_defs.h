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

#pragma once

#include <map>
#include <vector>
#include <utility>

namespace forte {

/// A structure to store how the string J is connected to the string I and the corresponding sign
/// I -> sign J
/// The uint32_t will hold up to 4,294,967,296 elements (that should be enough)
struct StringSubstitution {
    const double sign;
    const uint32_t I;
    const uint32_t J;
    StringSubstitution(const double& sign_, const uint32_t& I_, const uint32_t& J_)
        : sign(sign_), I(I_), J(J_) {}
};

/// 1-hole string substitution
struct H1StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const size_t J;
    H1StringSubstitution(int16_t sign_, int16_t p_, size_t J_) : sign(sign_), p(p_), J(J_) {}
};

/// 2-hole string substitution
struct H2StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const int16_t q;
    size_t J;
    H2StringSubstitution(int16_t sign_, int16_t p_, int16_t q_, size_t J_)
        : sign(sign_), p(p_), q(q_), J(J_) {}
};

/// 3-hole string substitution
struct H3StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const int16_t q;
    const int16_t r;
    const size_t J;
    H3StringSubstitution(int16_t sign_, int16_t p_, int16_t q_, int16_t r_, size_t J_)
        : sign(sign_), p(p_), q(q_), r(r_), J(J_) {}
};

using StringList = std::vector<std::vector<String>>;

/// Maps the integers (p,q,h) to list of strings connected by a^{+}_p a_q, where the string
/// I belongs to the irrep h
using VOList = std::map<std::tuple<size_t, size_t, int>, std::vector<StringSubstitution>>;

/// Maps the integers (class_I, class_J) to a map of orbital indices (p,q) and the corresponding
/// list of strings connected by a^{+}_p a_q, where the string I belongs to class_I and J belongs to
/// class_J
using VOListElement = std::map<std::tuple<int, int>, std::vector<StringSubstitution>>;
using VOListMap = std::map<std::pair<int, int>, VOListElement>;

/// Maps the integers (p,q,r,s,h) to list of strings connected by a^{+}_p a^{+}_q a_s a_r, where the
/// string I belongs to the irrep h
using VVOOList =
    std::map<std::tuple<size_t, size_t, size_t, size_t, int>, std::vector<StringSubstitution>>;
using VVOOListElement = std::map<std::tuple<int, int, int, int>, std::vector<StringSubstitution>>;
using VVOOListMap = std::map<std::pair<int, int>, VVOOListElement>;

/// Maps the integers (pq_sym, pq, h) to list of strings connected by a^{+}_p a^{+}_q a_q a_p where
/// the string I belongs to the irrep h
using OOList = std::map<std::tuple<int, size_t, int>, std::vector<StringSubstitution>>;
using OOListElement = std::map<std::tuple<int, int>, std::vector<uint32_t>>;
using OOListMap = std::map<int, OOListElement>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H1List = std::map<std::tuple<int, size_t, int>, std::vector<H1StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H2List = std::map<std::tuple<int, size_t, int>, std::vector<H2StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q a_r, where the
/// string I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H3List = std::map<std::tuple<int, size_t, int>, std::vector<H3StringSubstitution>>;

using Pair = std::pair<int, int>;
using PairList = std::vector<std::vector<std::pair<int, int>>>;

} // namespace forte
