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

#include <cstdint>
#include <memory>
#include <vector>

#include "determinant_common.h"

namespace psi {
namespace forte {

class FCIIntegrals;
class STLBitsetDeterminant;

class UI64Determinant {
  public:
    using bit_t = uint64_t;
    static constexpr int num_det_bits = 128;
    static constexpr int num_str_bits = 64;

    UI64Determinant();
    UI64Determinant(const STLBitsetDeterminant& d);
    explicit UI64Determinant(const std::vector<bool>& occupation);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit UI64Determinant(const std::vector<bool>& occupation_a,
                             const std::vector<bool>& occupation_b);

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
    void set_alfa_bits(bit_t x);
    void set_beta_bits(bit_t x);
    /// Return the value of a beta bit
    void zero_spin(DetSpinType spin_type);

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const;
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ(int norb) const;
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir(int norb) const;
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const;

    /// Set the value of an alpha bit
    double create_alfa_bit(int n);
    /// Set the value of a beta bit
    double create_beta_bit(int n);
    /// Set the value of an alpha bit
    double destroy_alfa_bit(int n);
    /// Set the value of a beta bit
    double destroy_beta_bit(int n);

    /// Return the sign of a_n applied to this determinant
    double slater_sign_a(int n) const;
    double slater_sign_aa(int n, int m) const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_b(int n) const;
    double slater_sign_bb(int n, int m) const;

    double slater_sign_aaaa(int i, int j, int a, int b) const;
    double slater_sign_bbbb(int i, int j, int a, int b) const;

    /// Count the number of alpha bits set to true
    int count_alfa() const;
    /// Count the number of beta bits set to true
    int count_beta() const;
    /// Return the number of alpha/beta pairs
    int npair() const;

    /// Flip all bits
    UI64Determinant& flip();

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

    /// Save the Slater determinant as a string
    std::string str( int n) const;

    struct Hash {
        std::size_t operator()(const psi::forte::UI64Determinant& bs) const {
            return bs.a_ * 31 + bs.b_;
        }
    };

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

double spin2(const UI64Determinant& lhs, const UI64Determinant& rhs);

///// XOR operator
//STLBitsetDeterminant operator^(const STLBitsetDeterminant& lhs) const;
///// XOR operator
//STLBitsetDeterminant& operator^=(const STLBitsetDeterminant& lhs);
///// &= operator
//STLBitsetDeterminant& operator&=(const STLBitsetDeterminant& lhs);
///// &= operator
//STLBitsetDeterminant& operator|=(const STLBitsetDeterminant& lhs);

/// Find the spin orbitals that are occupied in both determinants (performs a bitwise AND, &)
UI64Determinant common_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise XOR, ^)
UI64Determinant different_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
UI64Determinant union_occupation(const UI64Determinant& lhs, const UI64Determinant& rhs);


/// Given a set of determinant adds new elements necessary to have a spin complete set
void enforce_spin_completeness(std::vector<UI64Determinant>& det_space, int nmo);

}
}

#endif // _ui64_determinant_h_
