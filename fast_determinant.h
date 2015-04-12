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

#ifndef _fast_determinant_h_
#define _fast_determinant_h_

#include <cstdint>

#include "integrals.h"

namespace psi{ namespace libadaptive{

using bit_t = uint64_t;

/**
 * A class to store a Slater determinant using 64-bit unsigned integers.
 *
 * The determinant is represented by a pair of alpha/beta strings
 * that specify the occupation of each molecular orbital
 * (excluding frozen core and virtual orbitals).
 *
 * |Det> = |Ia> x |Ib>
 *
 * The strings are represented using an array of bits, and the
 * following convention is used here:
 * true <-> 1
 * false <-> 0
 */
class FastDeterminant{
public:
    // Class Constructor and Destructor
    /// Construct an empty determinant
    FastDeterminant() : nmo_(0),alfa_bits_(0), beta_bits_(0) {}

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit FastDeterminant(const std::vector<int>& occupation,bool print_det = false);

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit FastDeterminant(const std::vector<bool>& occupation,bool print_det = false);

    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit FastDeterminant(const std::vector<bool>& occupation_a,const std::vector<bool>& occupation_b,bool print_det = false);

    bool operator<(const FastDeterminant& lhs) const{
        if (alfa_bits_ > lhs.alfa_bits_) return false;
        if (alfa_bits_ < lhs.alfa_bits_) return true;
        return beta_bits_ < lhs.beta_bits_;
    }

    bool operator==(const FastDeterminant& lhs) const{
        return ((alfa_bits_ == lhs.alfa_bits_) and (beta_bits_ == lhs.beta_bits_));
    }

    void copy(const FastDeterminant &rhs)
    {
       alfa_bits_ = rhs.alfa_bits_;
       beta_bits_ = rhs.beta_bits_;
    }

    /// Get a pointer to the alpha bits
    bit_t alfa_bits() const {return alfa_bits_;}

    /// Get a pointer to the beta bits
    bit_t beta_bits() const {return beta_bits_;}

    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const {return (0 != (alfa_bits_ & (1UL << n)));}
    /// Return the value of a beta bit
    bool get_beta_bit(int n) const {return (0 != (beta_bits_ & (1UL << n)));}

    /// Set the value of an alpha bit
    void set_alfa_bit(int n, bool v) {alfa_bits_ ^= (-v ^ alfa_bits_) & (1 << n);}
    /// Set the value of a beta bit
    void set_beta_bit(int n, bool v) {beta_bits_ ^= (-v ^ beta_bits_) & (1 << n);}

    /// Specify the occupation numbers
    void set_alfa_bits(const bit_t alfa_bits) {alfa_bits_ = alfa_bits;}
    /// Specify the occupation numbers
    void set_beta_bits(const bit_t beta_bits) {beta_bits_ = beta_bits;}

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ();
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ();
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir();
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir();


    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ() const;
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ() const;
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir() const;
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir() const;

    /// Print the Slater determinant
    void print() const;
    /// Save the Slater determinant as a string
    std::string str() const;

    /// Compute the energy of a Slater determinant
    double energy() const;
    /// Compute the matrix element of the Hamiltonian between this determinant and a given one
    double slater_rules(const FastDeterminant& rhs) const;
    /// Compute the matrix element of the S^2 operator between this determinant and a given one
    double spin2(const FastDeterminant& rhs) const;

    /// Sets the pointer to the integral object
    static void set_ints(ExplorerIntegrals* ints) {
        ints_ = ints;
    }
private:
    // Data
    /// Number of non-frozen molecular orbitals
    unsigned short nmo_;
public:
    /// The occupation vector for the alpha electrons (does not include the frozen orbitals)
    bit_t alfa_bits_;
    /// The occupation vector for the beta electrons (does not include the frozen orbitals)
    bit_t beta_bits_;

    // Static data
    /// A pointer to the integral object
    static ExplorerIntegrals* ints_;
    static double SlaterSign(const bit_t I,int n);
};

//std::size_t hash_value(const FastDeterminant& input);


}} // End Namespaces

namespace std {
template <>
class hash<psi::libadaptive::FastDeterminant> {
public:
    size_t operator()(const psi::libadaptive::FastDeterminant &det) const
    {
        return det.alfa_bits_ ^ det.beta_bits_;
    }
};
}

#endif // _fast_determinant_h_
