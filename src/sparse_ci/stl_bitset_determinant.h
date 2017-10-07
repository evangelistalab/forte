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

#ifndef _stl_determinant_h_
#define _stl_determinant_h_

#include <bitset>
#include <unordered_map>

namespace psi {
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

class STLBitsetDeterminant {
  public:
    using bit_t = std::bitset<256>;
    // Class Constructor and Destructor
    /// Construct an empty determinant
    STLBitsetDeterminant();
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    STLBitsetDeterminant(const std::vector<int>& occupation, int nmo);
    STLBitsetDeterminant(const std::vector<int>& occupation);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    STLBitsetDeterminant(const std::vector<bool>& occupation, int nmo);
    STLBitsetDeterminant(const std::vector<bool>& occupation);
    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    STLBitsetDeterminant(const std::vector<bool>& occupation_a,
                         const std::vector<bool>& occupation_b);
    /// Construct a determinant from a bitset object
    STLBitsetDeterminant(const bit_t& bits);
    STLBitsetDeterminant(const bit_t& bits, int nmo);
    STLBitsetDeterminant(int nmo) { nmo_ = nmo; }
    STLBitsetDeterminant(const STLBitsetDeterminant& lhs);
    /// Construct a determinant from two STLBitsetStrings
    //    explicit STLBitsetDeterminant(const STLBitsetString& alpha, const STLBitsetString& beta);

    void copy(const STLBitsetDeterminant& rhs);

    /// Equal operator
    bool operator==(const STLBitsetDeterminant& lhs) const;
    /// Less than operator
    bool operator<(const STLBitsetDeterminant& lhs) const;
    /// XOR operator
    STLBitsetDeterminant operator^(const STLBitsetDeterminant& lhs) const;

    /// Get a pointer to the bits
    const bit_t& bits() const;

    const int nmo() const { return nmo_; }

    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const;
    /// Return the value of a beta bit
    bool get_beta_bit(int n) const;

    /// Set the value of an alpha bit
    void set_alfa_bit(int n, bool value);
    /// Set the value of a beta bit
    void set_beta_bit(int n, bool value);

    /// Switch the alpha and beta occupations
    void spin_flip();

    /// Return determinant with one spin zeroed, alpha == 0
    void zero_spin(bool spin);

    /// Get the alpha bits
    std::vector<bool> get_alfa_bits_vector_bool();
    /// Get the beta bits
    std::vector<bool> get_beta_bits_vector_bool();
    /// Get the alpha bits
    const std::vector<bool> get_alfa_bits_vector_bool() const;
    /// Get the beta bits
    const std::vector<bool> get_beta_bits_vector_bool() const;
    /// Return the number of alpha/beta pairs
    int npair();

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

    /// Set the value of an alpha bit
    double create_alfa_bit(int n);
    /// Set the value of a beta bit
    double create_beta_bit(int n);
    /// Set the value of an alpha bit
    double destroy_alfa_bit(int n);
    /// Set the value of a beta bit
    double destroy_beta_bit(int n);

    /// Print the Slater determinant
    void print() const;
    /// Save the Slater determinant as a string
    std::string str() const;

    /// Apply S+ to this determinant
    std::vector<std::pair<STLBitsetDeterminant, double>> spin_plus() const;
    /// Apply S- to this determinant
    std::vector<std::pair<STLBitsetDeterminant, double>> spin_minus() const;
    /// Compute the matrix element of the S^2 operator between this determinant
    /// and a given one
    double spin2_slow(const STLBitsetDeterminant& rhs) const;
    /// Return the eigenvalue of Sz
    double spin_z() const;
    /// Compute the matrix element of the S^2 operator between this determinant
    /// and a given one
    double spin2(const STLBitsetDeterminant& rhs) const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_a(int n) const;
    double slater_sign_aa(int n, int m) const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_b(int n) const;
    double slater_sign_bb(int n, int m) const;

    double slater_sign(int i, int j, int a, int b) const;

    /// Perform an alpha-alpha single excitation (i->a)
    double single_excitation_a(int i, int a);
    /// Perform an beta-beta single excitation (I -> A)
    double single_excitation_b(int i, int a);
    /// Perform an alpha-alpha double excitation (ij->ab)
    double double_excitation_aa(int i, int j, int a, int b);
    /// Perform an alpha-beta double excitation (iJ -> aB)
    double double_excitation_ab(int i, int j, int a, int b);
    /// Perform an beta-beta double excitation (IJ -> AB)
    double double_excitation_bb(int i, int j, int a, int b);

    /// The occupation vector (does not include the frozen orbitals)
    bit_t bits_;
    /// Number of non-frozen molecular orbitals
    int nmo_;

    /// Return the sign of a_n applied to string I
    static double SlaterSign(const bit_t& I, int n);
    double SlaterSign(int n);
    /// Return the sign of a_n^+ a_m applied to string I
    double SlaterSign(const bit_t& I, int m, int n);
    /// Return the sign of a_a^+ a_b^+ a_j a_i applied to string I
    double SlaterSign(const bit_t& bits, int i, int j, int a, int b);
    /// Given a set of determinant adds new elements necessary to have a spin
    /// complete set
    void enforce_spin_completeness(std::vector<STLBitsetDeterminant>& det_space);

    struct Hash {
        std::size_t operator()(const psi::forte::STLBitsetDeterminant& bs) const {
            return std::hash<bit_t>()(bs.bits_);
        }
    };
};

using Determinant = STLBitsetDeterminant;
using det_vec = std::vector<STLBitsetDeterminant>;
template <typename T = double>
using det_hash = std::unordered_map<STLBitsetDeterminant, T, STLBitsetDeterminant::Hash>;
using det_hash_it =
    std::unordered_map<STLBitsetDeterminant, double, STLBitsetDeterminant::Hash>::iterator;
}
} // End Namespaces

#endif // _bitset_determinant_h_
