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

#ifndef _sq_h_
#define _sq_h_

#include <vector>
#include <string>
#include <map>

namespace psi {
namespace forte {

/**
 * @brief The SqOperator class
 */
class SqOperator {
  public:
    /// Default constructor
    SqOperator();
    /// Initialize with vector with a list of creation and annihilation
    /// operators
    SqOperator(const std::vector<int>& cre, const std::vector<int>& ann);
    /// Return the number of creation operators
    size_t ncre() const { return cre_.size(); }
    /// Return the number of annihilation operators
    size_t nann() const { return ann_.size(); }
    /// Return the creation operators
    const std::vector<int>& cre() const { return cre_; }
    /// Return the annihilation operators
    const std::vector<int>& ann() const { return ann_; }
    /// Return a string representation of this object
    std::string str() const;
    /// A hash function
    bool operator==(const SqOperator& lhs) const;
    /// A hash function
    bool operator<(const SqOperator& lhs) const;
    /// A hash function
    std::size_t hash();
    /// Sort and return the sign of the permutation
    double sort();
    /// Test the sorting
    void test_sort();

  private:
    /// List of creation operators
    std::vector<int> cre_;
    /// List of annihilation operators
    std::vector<int> ann_;

    double sort(std::vector<int>& vec);
};

using op_hash = std::map<SqOperator, double>;

/**
 * @brief The Operator class
 */
class Operator {
  public:
    /// Default constructor
    Operator();
    /// Add an operator
    void add(double value, const SqOperator& op);
    /// Return a string representation of this object
    std::string str() const;
    /// Return the hash object
    const op_hash& ops();

  private:
    /// A vector that contains the components of the operator
    op_hash ops_;
};

/**
 * @brief The WickTheorem class
 */
class WickTheorem {
  public:
    /// Default constructor
    WickTheorem();
    /// Apply Wick's theorem to a product of operators
    Operator evaluate(Operator& lhs, Operator& rhs);

  private:
    /// Contract a pair of SqOperators in all possible ways
    void contract(const SqOperator& lhs, const SqOperator& rhs, Operator& res);
    /// Contract a pair of SqOperators using a fixed contraction pattern
    std::pair<double, SqOperator> simple_contract(const SqOperator& lhs, const SqOperator& rhs,
                                                  const std::vector<std::pair<int, int>>& pattern);
};

/**
 * @brief The SqTest class
 */
class SqTest {
  public:
    /// Default constructor
    SqTest();

  private:
};
}
} // End Namespaces

#endif // _sq_h_
