/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _configuration_h_
#define _configuration_h_

// #include <string>
// #include <vector>
// #include <iostream>

// #include "bitarray.hpp"
// #include "bitwise_operations.hpp"

namespace forte {

/**
 * @brief A class to represent a Configuration with N/2 spatial orbitals
 *
 * Spatial orbitals are divided into N/2 doubly occupied and N/2 singly occupied sets.
 * For each set we pack groups of 64 orbitals occupations into an array of
 * 64-bit unsigned integers. Each group of 64 bits is called a word.
 * Each set of N/2 bits is stored in K = ceil(N/2 / 64) words.
 * So the full determinant is stored as an array of words of the form
 * [word 1][word 2]...[word K][word K+1]...[word 2K]
 */
template <size_t N> class ConfigurationImpl : public BitArray<N> {
  public:
    // Since the template parent (BitArray) of this template class is not instantiated during the
    // compilation pass, here we declare all the member variables and functions inherited and used
    using BitArray<N>::nbits;
    // using BitArray<N>::bits_per_word;
    // using BitArray<N>::nwords_;
    using BitArray<N>::words_;
    // using BitArray<N>::count;
    using BitArray<N>::get_bit;
    using BitArray<N>::set_bit;
    using BitArray<N>::maskbit;
    // using BitArray<N>::whichbit;
    // using BitArray<N>::whichword;
    // using BitArray<N>::getword;
    // using BitArray<N>::slater_sign;
    // using BitArray<N>::operator|;
    // using BitArray<N>::operator^;
    // using BitArray<N>::operator&;
    // using BitArray<N>::fast_a_xor_b_count;
    // using BitArray<N>::fast_a_and_b_eq_zero;

    /// the number of bits divided by two
    static constexpr size_t nbits_half = N / 2;

    static_assert(N % 128 == 0,
                  "The number of bits in the Configuration class must be a multiple of 128");

    /// half of the words used to store the bits
    static constexpr size_t nwords_half = BitArray<N>::nwords_ / 2;

    /// the starting bit for singly orbital occupations
    static constexpr size_t socc_bit_offset = nwords_half * BitArray<N>::bits_per_word;

    /// returns the number of orbitals (half the number of bits)
    static constexpr size_t norb() { return nbits_half; }

    /// Default constructor
    ConfigurationImpl() : BitArray<N>() {}

    /// Constructor from a bit array
    ConfigurationImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit ConfigurationImpl(const DeterminantImpl<N>& d) {
        for (int p = 0; p < nbits_half; ++p) {
            set_occ(p, static_cast<int>(d.get_alfa_bit(p)) + static_cast<int>(d.get_beta_bit(p)));
        }
    }

    // Functions to set the occupation

    /// set the occupation of an orbital
    void set_occ(size_t pos, int val) {
        if (val == 0) {
            set_bit(pos, false);
            set_bit(pos + socc_bit_offset, false);
        } else if (val == 1) {
            set_bit(pos, false);
            set_bit(pos + socc_bit_offset, true);
        } else if (val == 2) {
            set_bit(pos, true);
            set_bit(pos + socc_bit_offset, false);
        }
    }
    /// is orbital pos empty?
    bool is_empt(size_t pos) const { return (not is_docc(pos)) and (not is_socc(pos)); }

    /// is orbital pos doubly occupied?
    bool is_docc(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[0] & maskbit(pos);
        } else {
            return get_bit(pos);
        }
    }

    /// is orbital pos singly occupied?
    bool is_socc(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[1] & maskbit(pos);
        } else {
            return get_bit(pos + socc_bit_offset);
        }
    }
};

// Functions

/**
 * @brief return a string representation of this configuration
 */
template <size_t N>
std::string str_bits(const ConfigurationImpl<N>& d, int n = ConfigurationImpl<N>::nbits) {
    std::string s;
    for (int p = 0; p < n; ++p) {
        s += d.get_bit(p) ? '1' : '0';
    }
    return s;
}

template <size_t N>
std::string str(const ConfigurationImpl<N>& d, int n = ConfigurationImpl<N>::nbits_half) {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (d.is_docc(p)) {
            s += "2";
        } else if (d.is_socc(p)) {
            s += "1";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

} // namespace forte

#endif // _determinant_hpp_
