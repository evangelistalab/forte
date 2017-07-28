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

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS public

#ifndef _dynamic_bitset_determinant_h_
#define _dynamic_bitset_determinant_h_

#define mix_fasthash(h)                                                                            \
    ({                                                                                             \
        (h) ^= (h) >> 23;                                                                          \
        (h) *= 0x2127599bf4325c37ULL;                                                              \
        (h) ^= (h) >> 47;                                                                          \
    })

#include "mini-boost/boost/dynamic_bitset.hpp"
#include <unordered_map>

#include "integrals/integrals.h"
#include "fci/fci_vector.h"

namespace psi {
namespace forte {

/**
 * A class to store a Slater determinant using Boost's dynamic_bitset.
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
class DynamicBitsetDeterminant {
  public:
    using bit_t = boost::dynamic_bitset<>;

    // test integrals
    void test_ints() { outfile->Printf("\n FC energy: %1.8f", fci_ints_->frozen_core_energy()); }

    // Class Constructor and Destructor
    /// Construct an empty determinant
    DynamicBitsetDeterminant();

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DynamicBitsetDeterminant(int nmo);

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DynamicBitsetDeterminant(const std::vector<int>& occupation, bool print_det = false);

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DynamicBitsetDeterminant(const std::vector<bool>& occupation, bool print_det = false);

    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DynamicBitsetDeterminant(const std::vector<bool>& occupation_a,
                                      const std::vector<bool>& occupation_b,
                                      bool print_det = false);

    bool operator<(const DynamicBitsetDeterminant& lhs) const {
        if (alfa_bits_ > lhs.alfa_bits_)
            return false;
        if (alfa_bits_ < lhs.alfa_bits_)
            return true;
        return beta_bits_ < lhs.beta_bits_;
    }

    bool operator==(const DynamicBitsetDeterminant& lhs) const {
        return ((alfa_bits_ == lhs.alfa_bits_) and (beta_bits_ == lhs.beta_bits_));
    }

    //    DynamicBitsetDeterminant& operator=(const DynamicBitsetDeterminant
    //    &rhs)
    //    {
    //       alfa_bits_ = rhs.alfa_bits_;
    //       beta_bits_ = rhs.beta_bits_;
    //       return *this;
    //    }

    void copy(const DynamicBitsetDeterminant& rhs) {
        nmo_ = rhs.nmo_;
        alfa_bits_ = rhs.alfa_bits_;
        beta_bits_ = rhs.beta_bits_;
    }

    /// Get a pointer to the alpha bits
    const bit_t& alfa_bits() const { return alfa_bits_; }

    /// Get a pointer to the beta bits
    const bit_t& beta_bits() const { return beta_bits_; }
    /// Get the alpha bits
    std::vector<bool> get_alfa_bits_vector_bool() {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n) {
            result.push_back(alfa_bits_[n]);
        }
        return result;
    }
    /// Get the beta bits
    std::vector<bool> get_beta_bits_vector_bool() {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n) {
            result.push_back(beta_bits_[n]);
        }
        return result;
    }
    /// Get the alpha bits
    const std::vector<bool> get_alfa_bits_vector_bool() const {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n) {
            result.push_back(alfa_bits_[n]);
        }
        return result;
    }
    /// Get the beta bits
    const std::vector<bool> get_beta_bits_vector_bool() const {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n) {
            result.push_back(beta_bits_[n]);
        }
        return result;
    }

    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const { return alfa_bits_[n]; }
    /// Return the value of a beta bit
    bool get_beta_bit(int n) const { return beta_bits_[n]; }

    /// Set the value of an alpha bit
    void set_alfa_bit(int n, bool value) { alfa_bits_[n] = value; }
    /// Set the value of a beta bit
    void set_beta_bit(int n, bool value) { beta_bits_[n] = value; }

    /// Specify the occupation numbers
    void set_alfa_bits(const bit_t& alfa_bits) { alfa_bits_ = alfa_bits; }
    /// Specify the occupation numbers
    void set_beta_bits(const bit_t& beta_bits) { beta_bits_ = beta_bits; }

    /// Switch the alpha and beta occupations
    void spin_flip();

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

    /// Compute the energy of a Slater determinant
    double energy() const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules(const DynamicBitsetDeterminant& rhs) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_alpha(int i, int a) const;
    /// Compute the matrix element of the Hamiltonian between this determinant
    /// and a given one
    double slater_rules_single_beta(int i, int a) const;
    /// Compute the matrix element of the S^2 operator between this determinant
    /// and a given one
    double spin2(const DynamicBitsetDeterminant& rhs) const;
    /// Apply S+ to this determinant
    std::vector<std::pair<DynamicBitsetDeterminant, double>> spin_plus() const;
    /// Apply S- to this determinant
    std::vector<std::pair<DynamicBitsetDeterminant, double>> spin_minus() const;
    /// Compute the matrix element of the S^2 operator between this determinant
    /// and a given one
    double spin2_slow(const DynamicBitsetDeterminant& rhs) const;
    /// Return the eigenvalue of Sz
    double spin_z() const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_alpha(int n) const;
    /// Return the sign of a_n applied to this determinant
    double slater_sign_beta(int n) const;
    /// Perform an alpha-alpha double excitation (ij->ab)
    double double_excitation_aa(int i, int j, int a, int b);
    /// Perform an alpha-beta double excitation (iJ -> aB)
    double double_excitation_ab(int i, int j, int a, int b);
    /// Perform an alpha-beta double excitation (IJ -> AB)
    double double_excitation_bb(int i, int j, int a, int b);

    /// Check if a space of determinants contains duplicates
    static void check_uniqueness(std::vector<DynamicBitsetDeterminant>);

    /// Sets the pointer to the integral object
    static void set_ints(std::shared_ptr<FCIIntegrals> ints);
    /// Resets the pointer to the integral object
    static void reset_ints();

  public:
    // Data
    /// Number of non-frozen molecular orbitals
    int nmo_;
    /// The occupation vector for the alpha electrons (does not include the
    /// frozen orbitals)
    bit_t alfa_bits_;
    /// The occupation vector for the beta electrons (does not include the
    /// frozen orbitals)
    bit_t beta_bits_;

    // Static data
    /// Precomputed bit masks of the form : 111...1000...0.  Used by
    /// FastSlaterSign.
    static std::vector<bit_t> bit_mask_;
    /// A pointer to the integral object
    static std::shared_ptr<FCIIntegrals> fci_ints_;
    /// Return the sign of a_n applied to string I
    static double SlaterSign(const bit_t& I, int n);
    /// Return the sign of a_n applied to string I
    static double FastSlaterSign(const boost::dynamic_bitset<>& I, int n);

    struct Hash {
        std::size_t operator()(const psi::forte::DynamicBitsetDeterminant& bs) const {
            size_t h = 0;
            for (int p = 0; p < bs.nmo_; p++) {
                if (bs.alfa_bits_[p]) {
                    h += (1 << p);
                }
                if (bs.beta_bits_[p]) {
                    h += (1 << (p + bs.nmo_));
                }
            }
            return h;
        }
    };
};
}
} // End Namespaces

#endif // _dynamic_bitset_determinant_h_
