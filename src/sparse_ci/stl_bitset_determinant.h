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

#ifndef MAX_DET_ORB
#define MAX_DET_ORB 128
#endif

#include <bitset>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include "determinant_common.h"


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
    /// The number of bits used to represent a string (half a determinant)
    static constexpr int num_str_bits = MAX_DET_ORB;
    /// The number of bits used to represent a determinant
    static constexpr int num_det_bits = 2 * num_str_bits;

    /// The bitset type
    using bit_t = std::bitset<num_det_bits>;

    // Class Constructor and Destructor

    /// Construct an empty determinant
    explicit STLBitsetDeterminant();
    STLBitsetDeterminant(int n) = delete;
    STLBitsetDeterminant(size_t n) = delete;
    /// Construct a determinant from a bitset object
    explicit STLBitsetDeterminant(const bit_t& bits);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit STLBitsetDeterminant(const std::vector<bool>& occupation);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit STLBitsetDeterminant(const std::vector<bool>& occupation_a,
                                  const std::vector<bool>& occupation_b);

    /// Return the bitset
    const bit_t& bits() const;

    /// Equal operator
    bool operator==(const STLBitsetDeterminant& lhs) const;
    /// Less than operator
    bool operator<(const STLBitsetDeterminant& lhs) const;

    /// Less than operator
    static bool less_than(const STLBitsetDeterminant& rhs, const STLBitsetDeterminant& lhs);
    /// Reverse string ordering
    static bool reverse_less_then(const STLBitsetDeterminant& i, const STLBitsetDeterminant& j);

    /// Flip all bits
    STLBitsetDeterminant& flip();

    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const;
    /// Return the value of a beta bit
    bool get_beta_bit(int n) const;

    /// Set the value of an alpha bit
    void set_alfa_bit(int n, bool value);
    /// Set the value of a beta bit
    void set_beta_bit(int n, bool value);

    /// Return determinant with one spin zeroed, alpha == 0
    void zero_spin(DetSpinType spin_type);

    /// Count the number of alpha bits set to true
    int count_alfa() const;
    /// Count the number of beta bits set to true
    int count_beta() const;
    /// Return the number of alpha/beta pairs
    int npair();

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const;
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ(int norb) const;
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir(int norb) const;
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const;

    /// Return a psi::Dimension object for occupied alpha orbitals
    std::vector<std::vector<int>> get_asym_occ(std::vector<int> act_mo) const;
    /// Return a psi::Dimension object for occupied beta orbital
    std::vector<std::vector<int>> get_bsym_occ(std::vector<int> act_mo) const;
    /// Return a psi::Dimension object for virtual alpha orbital
    std::vector<std::vector<int>> get_asym_vir(std::vector<int> act_mo) const;
    /// Return a psi::Dimension object for virtual beta orbitals
    std::vector<std::vector<int>> get_bsym_vir(std::vector<int> act_mo) const;

    //    /// Return a vector of occupied alpha orbitals
    //    std::vector<int> get_alfa_occ() const;
    //    /// Return a vector of occupied beta orbitals
    //    std::vector<int> get_beta_occ() const;
    //    /// Return a vector of virtual alpha orbitals
    //    std::vector<int> get_alfa_vir() const;
    //    /// Return a vector of virtual beta orbitals
    //    std::vector<int> get_beta_vir() const;

    /// Set the value of an alpha bit
    double create_alfa_bit(int n);
    /// Set the value of a beta bit
    double create_beta_bit(int n);
    /// Set the value of an alpha bit
    double destroy_alfa_bit(int n);
    /// Set the value of a beta bit
    double destroy_beta_bit(int n);

    /// Save the Slater determinant as a string
    std::string str(int n = num_str_bits) const;

    /// Apply S+ to this determinant
    std::vector<std::pair<STLBitsetDeterminant, double>> spin_plus() const;
    /// Apply S- to this determinant
    std::vector<std::pair<STLBitsetDeterminant, double>> spin_minus() const;
    /// Return the expectation value of S_z
    double spin_z() const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_a(int n) const;
    double slater_sign_aa(int n, int m) const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_b(int n) const;
    double slater_sign_bb(int n, int m) const;

    double slater_sign_aaaa(int i, int j, int a, int b) const;
    double slater_sign_bbbb(int i, int j, int a, int b) const;

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

    struct Hash {
        std::size_t operator()(const forte::STLBitsetDeterminant& bs) const {
            return std::hash<bit_t>()(bs.bits_);
        }
    };

  private:
    /// The bits
    bit_t bits_;
    /// A mask for the alpha bits
    const static bit_t alfa_mask;
    /// A mask for the beta bits
    const static bit_t beta_mask;
};

/// Find the spin orbitals that are occupied in both determinants (performs a bitwise AND, &)
STLBitsetDeterminant common_occupation(const STLBitsetDeterminant& lhs,
                                       const STLBitsetDeterminant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise XOR, ^)
STLBitsetDeterminant different_occupation(const STLBitsetDeterminant& lhs,
                                          const STLBitsetDeterminant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
STLBitsetDeterminant union_occupation(const STLBitsetDeterminant& lhs,
                                      const STLBitsetDeterminant& rhs);

/// Given a set of determinant adds new elements necessary to have a spin complete set
void enforce_spin_completeness(std::vector<STLBitsetDeterminant>& det_space, int nmo);
/// Compute the matrix element of the S^2 operator between two determinants
double spin2(const STLBitsetDeterminant& lhs, const STLBitsetDeterminant& rhs);
}
} // End Namespaces

#endif // _bitset_determinant_h_
