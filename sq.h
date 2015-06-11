/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _sq_h_
#define _sq_h_

#include <vector>
#include <string>

namespace psi{ namespace libadaptive{

/**
 * @brief The SqOperator class
 */
class SqOperator
{
public:
    /// Default constructor
    SqOperator();
    /// Initialize with vector
    SqOperator(const std::vector<int>& cre,const std::vector<int>& ann);
    /// Return a string representation of this object
    std::string str() const;
    /// A hash function
    bool operator==(const SqOperator& lhs) const;
    /// A hash function
    bool operator<(const SqOperator& lhs) const;
    /// A hash function
    std::size_t hash();
    /// Sort and return the sign of the permutation
    bool sort();
    /// Test the sorting
    void test_sort();
private:
    /// List of creation operators
    std::vector<int> cre_;
    /// List of annihilation operators
    std::vector<int> ann_;

    bool sort(std::vector<int>& vec);
};


using op_hash = std::map<SqOperator,double>;

/**
 * @brief The Operator class
 */
class Operator
{
public:
    /// Default constructor
    Operator();
    /// Add an operator
    void add(double value,const SqOperator& op);
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
class WickTheorem
{
public:
    /// Default constructor
    WickTheorem();
    /// Apply Wick's theorem to a product of operators
    Operator evaluate(Operator& lhs,Operator& rhs);
private:
};


/**
 * @brief The SqTest class
 */
class SqTest
{
public:
    /// Default constructor
    SqTest();
private:
};

}} // End Namespaces

#endif // _sq_h_
