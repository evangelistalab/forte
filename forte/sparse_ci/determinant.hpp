/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER,
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

#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "bitarray.hpp"
#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

enum class DetSpinType { Alpha, Beta };

/**
 * @brief A class to represent a Slater determinant with N spin orbitals
 *
 * Spin orbitals are divided into N/2 alpha and N/2 beta set.
 * For each set we pack groups of 64 spin orbitals into an array of
 * 64-bit unsigned integers. Each group of 64 bits is called a word.
 * Each set of N/2 bits is stored in K = ceil(N/2 / 64) words.
 * So the full determinant is stored as an array of words of the form
 * [word 1][word 2]...[word K][word K+1]...[word 2K]
 */
template <size_t N> class DeterminantImpl : public BitArray<N> {
  public:
    // Since the template parent (BitArray) of this template class is not instantiated during the
    // compilation pass, here we declare all the member variables and functions inherited and used
    using BitArray<N>::nbits;
    using BitArray<N>::bits_per_word;
    using BitArray<N>::nwords_;
    using BitArray<N>::words_;
    using BitArray<N>::count;
    using BitArray<N>::get_bit;
    using BitArray<N>::set_bit;
    using BitArray<N>::maskbit;
    using BitArray<N>::whichbit;
    using BitArray<N>::whichword;
    using BitArray<N>::getword;
    using BitArray<N>::slater_sign;
    using BitArray<N>::operator|;
    using BitArray<N>::operator^;
    using BitArray<N>::operator&;
    using BitArray<N>::fast_a_xor_b_count;
    using BitArray<N>::fast_a_and_b_eq_zero;
    using BitArray<N>::find_first_one;
    using BitArray<N>::find_last_one;
    using BitArray<N>::zero;

    /// the number of bits divided by two
    static constexpr size_t nbits_half = N / 2;

    static_assert(N % 128 == 0,
                  "The number of bits in the Determinant class must be a multiple of 128");

    /// half of the words used to store the bits
    static constexpr size_t nwords_half = BitArray<N>::nwords_ / 2;

    /// the starting bit for beta orbitals
    static constexpr size_t beta_bit_offset = nwords_half * BitArray<N>::bits_per_word;

    /// returns the number of orbitals (half the number of bits)
    static constexpr size_t norb() { return nbits_half; }

    /// Default constructor
    DeterminantImpl() : BitArray<N>() {}

    /// Constructor from a bit array
    DeterminantImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DeterminantImpl(const std::vector<bool>& occupation_a,
                             const std::vector<bool>& occupation_b) {
        int a_size = occupation_a.size();
        for (int p = 0; p < a_size; ++p)
            set_alfa_bit(p, occupation_a[p]);
        int b_size = occupation_b.size();
        for (int p = 0; p < b_size; ++p)
            set_beta_bit(p, occupation_b[p]);
    }

    /// Construct the determinant from two initializer lists that specify which orbitals are
    /// occupied in the alpha and beta strings.  alfa_list = [Ia] and beta_list = [Ib]
    explicit DeterminantImpl(std::initializer_list<size_t> alfa_list,
                             std::initializer_list<size_t> beta_list) {
        for (auto i : alfa_list) {
            set_alfa_bit(i, true);
        }
        for (auto i : beta_list) {
            set_beta_bit(i, true);
        }
    }

    explicit DeterminantImpl(std::vector<size_t> alfa_list, std::vector<size_t> beta_list) {
        for (auto i : alfa_list) {
            set_alfa_bit(i, true);
        }
        for (auto i : beta_list) {
            set_beta_bit(i, true);
        }
    }

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DeterminantImpl(const BitArray<nbits_half>& Ia, const BitArray<nbits_half>& Ib) {
        this->set_str(Ia, Ib);
    }

    /// String constructor. Convert a std::string to a determinant.
    /// E.g. DeterminantImpl<64>("0011") gives the determinant|0011>
    DeterminantImpl(const std::string& str) { set_str(*this, str); }

    // Functions to access bits

    /// get the value of alfa bit pos
    bool get_alfa_bit(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[0] & maskbit(pos);
        } else {
            return get_bit(pos);
        }
    }

    /// get the value of beta bit pos
    bool get_beta_bit(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[1] & maskbit(pos);
        } else {
            return get_bit(pos + beta_bit_offset);
        }
    }

    // Functions to set bits

    /// set alfa bit pos to val
    void set_alfa_bit(size_t pos, bool val) { set_bit(pos, val); }

    /// set beta bit pos to val
    void set_beta_bit(size_t pos, bool val) { set_bit(pos + beta_bit_offset, val); }

    /// set the alpha/beta strings
    void set_str(const BitArray<nbits_half>& sa, const BitArray<nbits_half>& sb) {
        for (size_t n = 0; n < nwords_half; n++) {
            words_[n] = sa.get_word(n);
            words_[n + nwords_half] = sb.get_word(n);
        }
    }

    void set_alfa_str(const BitArray<nbits_half>& sa) {
        for (size_t n = 0; n < nwords_half; n++) {
            words_[n] = sa.get_word(n);
        }
    }

    void set_beta_str(const BitArray<nbits_half>& sb) {
        for (size_t n = 0; n < nwords_half; n++) {
            words_[n + nwords_half] = sb.get_word(n);
        }
    }

    void set(std::initializer_list<size_t> alfa_list, std::initializer_list<size_t> beta_list) {
        zero();
        for (auto i : alfa_list) {
            set_alfa_bit(i, true);
        }
        for (auto i : beta_list) {
            set_beta_bit(i, true);
        }
    }

    // Comparison operators
    static bool less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
        return rhs < lhs;
    }

    static bool reverse_less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
        if constexpr (nbits == 128) {
            return (rhs.words_[0] < lhs.words_[0]) or
                   ((rhs.words_[0] == lhs.words_[0]) and (rhs.words_[1] < lhs.words_[1]));
        } else {
            for (size_t n = nwords_half; n > 0;) {
                --n;
                if (rhs.words_[n] > lhs.words_[n])
                    return false;
                if (rhs.words_[n] < lhs.words_[n])
                    return true;
            }
            for (size_t n = nwords_; n > nwords_half + 1;) {
                --n;
                if (rhs.words_[n] > lhs.words_[n])
                    return false;
                if (rhs.words_[n] < lhs.words_[n])
                    return true;
            }
            return rhs.words_[nwords_half] < lhs.words_[nwords_half];
        }
    }

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const {
        std::vector<int> occ;
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error(
                "Determinant::get_alfa_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of alpha orbitals (" +
                std::to_string(nbits_half) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (get_alfa_bit(p)) {
                occ.push_back(p);
            }
        }
        return occ;
    }

    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ(int norb) const {
        std::vector<int> occ;
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error(
                "Determinant::get_beta_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of beta orbitals (" +
                std::to_string(nbits_half) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (get_beta_bit(p)) {
                occ.push_back(p);
            }
        }
        return occ;
    }

    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir(int norb) const {
        std::vector<int> vir;
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error(
                "Determinant::get_alfa_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of alpha orbitals (" +
                std::to_string(nbits_half) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (not get_alfa_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const {
        std::vector<int> vir;
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error(
                "Determinant::get_beta_occ(int norb) was passed a value of norb (" +
                std::to_string(norb) +
                "), which is larger than the maximum number of beta orbitals (" +
                std::to_string(nbits_half) + ").");
        }
        for (int p = 0; p < norb; ++p) {
            if (not get_beta_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Apply the alpha creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_alfa_bit(int n) {
        if (get_alfa_bit(n))
            return 0.0;
        set_alfa_bit(n, true);
        return slater_sign_a(n);
    }

    /// Apply the beta creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_beta_bit(int n) {
        if (get_beta_bit(n))
            return 0.0;
        set_beta_bit(n, true);
        return slater_sign_b(n);
    }

    /// Apply the alpha annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_alfa_bit(int n) {
        if (not get_alfa_bit(n))
            return 0.0;
        set_alfa_bit(n, false);
        return slater_sign_a(n);
    }

    /// Apply the beta annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_beta_bit(int n) {
        if (not get_beta_bit(n))
            return 0.0;
        set_beta_bit(n, false);
        return slater_sign_b(n);
    }

    /// Return the sign for a single second quantized operator
    /// This function ignores if bit n is set or not
    double slater_sign_a(int n) const {
        if constexpr (nbits == 128) {
            // specialization for 64 + 64 bits
            return ui64_sign(words_[0], n);
        } else {
            return slater_sign(n);
        }
    }

    /// Return the sign for a single second quantized operator
    /// This function ignores if bit n is set or not
    double slater_sign_b(int n) const {
        if constexpr (nbits == 128) {
            // specialization for 64 + 64 bits
            return ui64_sign(words_[1], n) * ui64_bit_parity(words_[0]);
        } else {
            return slater_sign(n + beta_bit_offset);
        }
    }

    /// Return the sign for a pair of alpha second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    /// n and m are not assumed to have any specific order
    double slater_sign_aa(int n, int m) const {
        if constexpr (nbits == 128) {
            // specialization for 64 + 64 bits
            return ui64_sign(words_[0], n, m);
        } else {
            return slater_sign(n, m);
        }
    }

    /// Return the sign for a pair of beta second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    /// n and m are not assumed to have any specific order
    double slater_sign_bb(int n, int m) const {
        if constexpr (nbits == 128) {
            // specialization for 64 + 64 bits
            return ui64_sign(words_[1], n, m);
        } else {
            return slater_sign(n + beta_bit_offset, m + beta_bit_offset);
        }
    }

    /// Return the sign for a quadruplet of alpha second quantized operators
    /// a+(a) a+(b) a(j) a(i) applied to this determinant
    /// Untested, needs documentation
    double slater_sign_aaaa(int i, int j, int a, int b) const {
        if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
            (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_aa(i, j) * slater_sign_aa(a, b));
            } else {
                return (slater_sign_aa(i, j) * slater_sign_aa(a, b));
            }
        } else {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_aa(i, b) * slater_sign_aa(j, a));
            } else {
                return (slater_sign_aa(i, a) * slater_sign_aa(j, b));
            }
        }
    }

    /// Return the sign for a quadruplet of beta second quantized operators
    /// a+(A) a+(B) a(J) a(I) applied to this determinant
    /// Untested, needs documentation
    double slater_sign_bbbb(int i, int j, int a, int b) const {
        if ((((i < a) && (j < a) && (i < b) && (j < b)) == true) ||
            (((i < a) || (j < a) || (i < b) || (j < b)) == false)) {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_bb(i, j) * slater_sign_bb(a, b));
            } else {
                return (slater_sign_bb(i, j) * slater_sign_bb(a, b));
            }
        } else {
            if ((i < j) ^ (a < b)) {
                return (-1.0 * slater_sign_bb(i, b) * slater_sign_bb(j, a));
            } else {
                return (slater_sign_bb(i, a) * slater_sign_bb(j, b));
            }
        }
    }

    /// Count the number of beta bits set to 1
    int count_alfa() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return ui64_bit_count(words_[0]);
        } else if (N == 256) {
            return ui64_bit_count(words_[0]) + ui64_bit_count(words_[1]);
        } else {
            return count(0, nwords_half);
        }
    }

    /// Count the number of beta bits set to 1
    int count_beta() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return ui64_bit_count(words_[1]);
        } else if (N == 256) {
            return ui64_bit_count(words_[2]) + ui64_bit_count(words_[3]);
        } else {
            return count(nwords_half, nwords_);
        }
    };

    /// Compares a subset of the words of this determinant with the words of another determinant
    bool equal(const size_t start, const size_t end, const DeterminantImpl<N>& d) const {
        for (size_t n = start; n < end; n++) {
            if (words_[n] != d.words_[n])
                return false;
        }
        return true;
    }

    /// Compares the alpha part of this determinant with the alpha part of another determinant
    bool equal_alfa(const DeterminantImpl<N>& d) const {
        if constexpr (N == 128) {
            return words_[0] == d.words_[0];
        } else if constexpr (N == 256) {
            return words_[0] == d.words_[0] and words_[1] == d.words_[1];
        } else {
            return equal(0, nwords_half, d);
        }
    }

    /// Compares the beta part of this determinant with the beta part of another determinant
    bool equal_beta(const DeterminantImpl<N>& d) const {
        if constexpr (N == 128) {
            return words_[1] == d.words_[1];
        } else if constexpr (N == 256) {
            return words_[2] == d.words_[2] and words_[3] == d.words_[3];
        } else {
            return equal(nwords_half, nwords_, d);
        }
    }

    /// Find the index of the first alpha bit set to 1
    uint64_t find_first_one_alfa() const { return find_first_one(0, nwords_half); }

    /// Find the index of the first beta bit set to 1 (spatial orbital index)
    uint64_t find_first_one_beta() const {
        if (auto res = find_first_one(nwords_half, nwords_); res != ~uint64_t(0))
            return res - norb();
        return ~uint64_t(0);
    }

    uint64_t find_last_one_alfa() const { return find_last_one(0, nwords_half); }

    uint64_t find_last_one_beta() const {
        if (auto res = find_last_one(nwords_half, nwords_); res != ~uint64_t(0))
            return res - norb();
        return ~uint64_t(0);
    }

    /// Return the number of alpha/beta pairs
    int npair() const {
        int count = 0;
        for (size_t k = 0; k < nwords_half; ++k) {
            count += ui64_bit_count(words_[k] & words_[k + nwords_half]);
        }
        return count;
    }

    /// Perform an alpha-alpha single excitation (i->a)
    /// assuming that i is occupied and a is empty
    double single_excitation_a(int i, int a) {
        set_alfa_bit(i, false);
        set_alfa_bit(a, true);
        return slater_sign_aa(i, a);
    }

    /// Perform an beta-beta single excitation (I -> A)
    /// assuming that i is occupied and a is empty
    double single_excitation_b(int i, int a) {
        set_beta_bit(i, false);
        set_beta_bit(a, true);
        return slater_sign_bb(i, a);
    }

    /// Perform an alpha-alpha double excitation (ij->ab)
    /// assuming that ij are occupied and ab are empty
    double double_excitation_aa(int i, int j, int a, int b) {
        set_alfa_bit(i, false);
        set_alfa_bit(j, false);
        set_alfa_bit(b, true);
        set_alfa_bit(a, true);
        return slater_sign_aaaa(i, j, a, b);
    }

    /// Perform an alpha-beta double excitation (iJ -> aB)
    /// /// assuming that ij are occupied and ab are empty
    double double_excitation_ab(int i, int j, int a, int b) {
        set_alfa_bit(i, false);
        set_beta_bit(j, false);
        set_beta_bit(b, true);
        set_alfa_bit(a, true);
        return slater_sign_aa(i, a) * slater_sign_bb(j, b);
    }

    /// Perform an beta-beta double excitation (IJ -> AB)
    double double_excitation_bb(int i, int j, int a, int b) {
        set_beta_bit(i, false);
        set_beta_bit(j, false);
        set_beta_bit(b, true);
        set_beta_bit(a, true);
        return slater_sign_bbbb(i, j, a, b);
    }

    BitArray<nbits_half> get_alfa_bits() const {
        BitArray<nbits_half> s;
        for (size_t i = 0; i < nwords_half; i++) {
            s.set_word(i, words_[i]);
        }
        return s;
    }

    BitArray<nbits_half> get_beta_bits() const {
        BitArray<nbits_half> s;
        for (size_t i = 0; i < nwords_half; i++) {
            s.set_word(i, words_[nwords_half + i]);
        }
        return s;
    }

    void copy_beta_bits(BitArray<nbits_half>& ba) const {
        if constexpr (N == 128) {
            ba.set_word(0, words_[1]);
        } else if constexpr (N == 256) {
            ba.set_word(0, words_[2]);
            ba.set_word(1, words_[3]);
        } else {
            for (size_t i = 0; i < nwords_half; i++) {
                ba.set_word(i, words_[nwords_half + i]);
            }
        }
    }

    BitArray<nbits_half> get_bits(DetSpinType spin_type) const {
        return (spin_type == DetSpinType::Alpha ? get_alfa_bits() : get_beta_bits());
    }

    /// Zero the alpha part of a determinant
    void zero_alfa() {
        for (size_t n = 0; n < nwords_half; n++)
            words_[n] = u_int64_t(0);
    }

    /// Zero the alpha part of a determinant
    void zero_beta() {
        for (size_t n = nwords_half; n < nwords_; n++)
            words_[n] = u_int64_t(0);
    }

    /// Swap the alpha and beta bits of a determinant
    DeterminantImpl<N> spin_flip() const {
        DeterminantImpl<N> d(*this);
        for (size_t n = 0; n < nwords_half; n++) {
            std::swap(d.words_[n], d.words_[n + nwords_half]);
        }
        return d;
    }

    /// Describe the excitation connection of a determinant relative to this one
    /// The excitation connection is a vector of 4 vectors:
    /// [[alfa holes], [alfa particles], [beta holes], [beta particles]]
    std::vector<std::vector<size_t>> excitation_connection(const DeterminantImpl<N>& d) const {
        std::vector<std::vector<size_t>> excitation(4);
        for (size_t i = 0; i < nbits_half; i++) {
            if (get_alfa_bit(i) and not d.get_alfa_bit(i)) {
                excitation[0].push_back(i);
            }
            if (not get_alfa_bit(i) and d.get_alfa_bit(i)) {
                excitation[1].push_back(i);
            }
            if (get_beta_bit(i) and not d.get_beta_bit(i)) {
                excitation[2].push_back(i);
            }
            if (not get_beta_bit(i) and d.get_beta_bit(i)) {
                excitation[3].push_back(i);
            }
        }
        return excitation;
    }
};

// Functions

/**
 * @brief return a string representation of this determinant
 */
template <size_t N>
std::string str_bits(const DeterminantImpl<N>& d, int n = DeterminantImpl<N>::nbits) {
    std::string s;
    for (int p = 0; p < n; ++p) {
        s += d.get_bit(p) ? '1' : '0';
    }
    return s;
}

template <size_t N>
std::string str(const DeterminantImpl<N>& d, int n = DeterminantImpl<N>::nbits_half) {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (d.get_alfa_bit(p) and d.get_beta_bit(p)) {
            s += "2";
        } else if (d.get_alfa_bit(p) and not d.get_beta_bit(p)) {
            s += "+";
        } else if (not d.get_alfa_bit(p) and d.get_beta_bit(p)) {
            s += "-";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

template <size_t N> std::ostream& operator<<(std::ostream& os, const DeterminantImpl<N>& d) {
    os << str(d);
    return os;
}

template <size_t N> void set_str(DeterminantImpl<N>& d, const std::string& str) {
    // zero all the bits and set the bits passed as a string
    d.zero();
    if (str.size() <= DeterminantImpl<N>::nbits) {
        size_t k = 0;
        for (auto c : str) {
            if (c == '0') {
                d.set_bit(k, 0);
            } else {
                d.set_bit(k, 1);
            }
            k++;
        }
    } else {
        throw std::range_error("template <size_t N> void set_str(DeterminantImpl<N>&, const "
                               "std::string&)\nmismatch "
                               "between the number of bits in d and those passed in str\n");
    }
}

template <size_t N>
std::vector<std::vector<int>> get_asym_occ(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {

    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (d.get_alfa_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_bsym_occ(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (d.get_beta_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_asym_vir(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not d.get_alfa_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

template <size_t N>
std::vector<std::vector<int>> get_bsym_vir(const DeterminantImpl<N>& d,
                                           const std::vector<int>& act_mo) {
    size_t nirrep = act_mo.size();
    std::vector<std::vector<int>> occ(nirrep);

    int abs = 0;
    for (size_t h = 0; h < nirrep; ++h) {
        for (int p = 0; p < act_mo[h]; ++p) {
            if (not d.get_beta_bit(abs)) {
                occ[h].push_back(abs);
            }
            abs++;
        }
    }
    return occ;
}

/**
 * @brief Apply a general excitation operator to this determinant
 *        Details:
 *        (bc)_n ... (bc)_1 (ba)_n ... (ba)_1 (ac)_n ... (ac)_1 (aa)_n ... (aa)_1 |det>
 *        where aa = alpha annihilation operator
 *        where ac = alpha creation operator
 *        where ba = alpha annihilation operator
 *        where bc = alpha creation operator
 * @param aann list of alpha orbitals to annihilate
 * @param acre list of alpha orbitals to create
 * @param bann list of beta orbitals to annihilate
 * @param bcre list of beta orbitals to create
 * @return the sign of the final determinant (+1, -1, or 0)
 */
template <size_t N>
double gen_excitation(DeterminantImpl<N>& d, const std::vector<int>& aann,
                      const std::vector<int>& acre, const std::vector<int>& bann,
                      const std::vector<int>& bcre) {
    double sign = 1.0;
    for (auto i : aann) {
        sign *= d.slater_sign_a(i) * d.get_alfa_bit(i);
        d.set_alfa_bit(i, false);
    }
    for (auto i : acre) {
        sign *= d.slater_sign_a(i) * (1 - d.get_alfa_bit(i));
        d.set_alfa_bit(i, true);
    }
    for (auto i : bann) {
        sign *= d.slater_sign_b(i) * d.get_beta_bit(i);
        d.set_beta_bit(i, false);
    }
    for (auto i : bcre) {
        sign *= d.slater_sign_b(i) * (1 - d.get_beta_bit(i));
        d.set_beta_bit(i, true);
    }
    return sign;
}

template <size_t N>
double apply_operator_to_det(DeterminantImpl<N>& d, const DeterminantImpl<N>& cre,
                             const DeterminantImpl<N>& ann) {
    // loop over the annihilation operators (in ascending order)
    DeterminantImpl<N> temp(ann); // temp is for bookkeeping
    size_t n = temp.count();
    double sign = 1.0;
    for (size_t i = 0; i < n; ++i) {
        // find the next annihilation operator
        const uint64_t orb = temp.find_and_clear_first_one();
        // if this bit is set
        if (d.get_bit(orb) == 1) {
            // compute the sign
            sign *= d.slater_sign(orb);
            // set the bit to zero
            d.set_bit(orb, false);
        } else {
            return 0.0;
        }
    }
    // loop over the creation operators (in ascending order)
    temp = cre;
    n = temp.count();
    for (size_t i = 0; i < n; ++i) {
        // find the next creation operator
        const uint64_t orb = temp.find_and_clear_first_one();
        // if this bit is unset
        if (d.get_bit(orb) == 0) {
            // compute the sign
            sign *= d.slater_sign(orb);
            // set the bit to zero
            d.set_bit(orb, true);
        } else {
            return 0.0;
        }
    }
    // the creation operators are applied in the opposite order of the way
    // they are supposed to be applied (we should apply them in descending order).
    // this factor keeps into account the permutation sign for
    // reversing the order of the creation operators.
    sign *= 1.0 - 2.0 * ((n / 2) % 2);
    return sign;
}

///
/// So there are no checks in place
/// @brief Apply a general operator to this determinant. This function assumes we can apply this
/// operator to the determinant and should be used only after faster_can_apply_operator has been
/// used to check if the operator can be applied to the determinant.
/// @param d the determinant
/// @param new_d the new determinant
/// @param cre the creation operator
/// @param ann the annihilation operator
/// @param sign the sign mask (precomputed by the user) of the operator
/// @return the sign of the final determinant (+1, -1)
///
/// @note This function is faster than apply_operator_to_det
/// Example:
///
///   Determinant det, new_det, cre, ann, sign_mask, idx;
///   // test if the operator can be applied
///   if (det.faster_can_apply_operator(cre,ann)) {
///       // compute the sign mask
///       compute_sign_mask(cre, ann, sign_mask, idx);
///       auto value = faster_apply_operator_to_det(det, new_det, cre, ann, sign_mask);
///       // do something with value and new_det
///   }
///
template <size_t N>
inline double faster_apply_operator_to_det(const DeterminantImpl<N>& d, DeterminantImpl<N>& new_d,
                                           const DeterminantImpl<N>& cre,
                                           const DeterminantImpl<N>& ann,
                                           const DeterminantImpl<N>& sign) {
    size_t n = 0;
    if constexpr (N == 128) {
        // specialization for 64 + 64 bits
        new_d.words_[0] = d.words_[0] & (~ann.words_[0]);
        new_d.words_[1] = d.words_[1] & (~ann.words_[1]);
        n += ui64_bit_count(new_d.words_[0] & sign.words_[0]);
        n += ui64_bit_count(new_d.words_[1] & sign.words_[1]);
        new_d.words_[0] |= cre.words_[0];
        new_d.words_[1] |= cre.words_[1];
    } else if constexpr (N == 256) {
        new_d.words_[0] = d.words_[0] & (~ann.words_[0]);
        new_d.words_[1] = d.words_[1] & (~ann.words_[1]);
        new_d.words_[2] = d.words_[2] & (~ann.words_[2]);
        new_d.words_[3] = d.words_[3] & (~ann.words_[3]);
        n += ui64_bit_count(new_d.words_[0] & sign.words_[0]);
        n += ui64_bit_count(new_d.words_[1] & sign.words_[1]);
        n += ui64_bit_count(new_d.words_[2] & sign.words_[2]);
        n += ui64_bit_count(new_d.words_[3] & sign.words_[3]);
        new_d.words_[0] |= cre.words_[0];
        new_d.words_[1] |= cre.words_[1];
        new_d.words_[2] |= cre.words_[2];
        new_d.words_[3] |= cre.words_[3];
    } else {
        // loop over the words
        for (size_t i = 0; i < DeterminantImpl<N>::nwords_; ++i) {
            // apply the annihilation operator
            new_d.words_[i] = d.words_[i] & (~ann.words_[i]);
            // compute the sign
            n += ui64_bit_count(new_d.words_[i] & sign.words_[i]);
            // apply the creation operator
            new_d.words_[i] |= cre.words_[i];
        }
    }
    return 1.0 - 2.0 * (n & 1);
}

template <size_t N> double spin2(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
    int nmo = DeterminantImpl<N>::nbits_half;
    const DeterminantImpl<N>& I = lhs;
    const DeterminantImpl<N>& J = rhs;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    // Make sure that Ms is the same otherwise the matrix element is automatically zero
    if ((lhs.count_alfa() != rhs.count_alfa()) or (lhs.count_beta() != rhs.count_beta())) {
        return 0.0;
    }

    DeterminantImpl<N> lr_diff = lhs ^ rhs;

    int nadiff = lr_diff.count_alfa() / 2;
    int nbdiff = lr_diff.count_beta() / 2;
    int na = lhs.count_alfa();
    int nb = lhs.count_beta();
    int npair = lhs.npair();

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta - Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here is a bit complex
        for (int p = 0; p < nmo; ++p) {
            if (J.get_alfa_bit(p) and I.get_beta_bit(p) and (not J.get_beta_bit(p)) and
                (not I.get_alfa_bit(p)))
                i = p;
            if (J.get_beta_bit(p) and I.get_alfa_bit(p) and (not J.get_alfa_bit(p)) and
                (not I.get_beta_bit(p)))
                j = p;
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign = rhs.slater_sign_a(i) * rhs.slater_sign_b(j) * lhs.slater_sign_a(j) *
                          lhs.slater_sign_b(i);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
}

} // namespace forte
