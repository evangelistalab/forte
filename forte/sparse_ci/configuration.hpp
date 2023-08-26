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
    using BitArray<N>::nwords_;
    using BitArray<N>::words_;
    using BitArray<N>::count;
    using BitArray<N>::get_bit;
    using BitArray<N>::set_bit;
    using BitArray<N>::maskbit;

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

    /// Construct a configuration from a determinant object
    explicit ConfigurationImpl(const DeterminantImpl<N>& d) {
        for (size_t k = 0; k < nwords_half; ++k) {
            words_[k] = d.words_[k] & d.words_[k + nwords_half];
            words_[k + nwords_half] = d.words_[k] ^ d.words_[k + nwords_half];
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

    /// Count the number of doubly occupied orbitals
    int count_docc() const { return count(0, nwords_half); }

    /// Count the number of singly occupied orbitals
    int count_socc() const { return count(nwords_half, nwords_); }

    BitArray<nbits_half> get_docc_str() const {
        BitArray<nbits_half> str;
        for (size_t k = 0; k < nwords_half; ++k) {
            str.set_word(k, words_[k]);
        }
        return str;
    }

    /// Return a vector with the indices of the doubly occupied orbitals
    /// @param docc_vec is a vector large enough to contain the list of orbitals
    void get_docc_vec(int norb, std::vector<int>& docc_vec) const {
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error("Configuration::get_docc_vec(...) was passed a value of norb (" +
                                   std::to_string(norb) +
                                   "), which is larger than the maximum number of orbitals (" +
                                   std::to_string(nbits_half) + ").");
        }
        int k = 0;
        for (int p = 0; p < norb; ++p) {
            if (is_docc(p)) {
                docc_vec[k] = p;
                k++;
            }
        }
    }

    /// Return a vector with the indices of the singly occupied orbitals
    /// @param socc_vec is a vector large enough to contain the list of orbitals
    void get_socc_vec(int norb, std::vector<int>& socc_vec) const {
        if (static_cast<size_t>(norb) > nbits_half) {
            throw std::range_error("Configuration::get_socc_vec(...) was passed a value of norb (" +
                                   std::to_string(norb) +
                                   "), which is larger than the maximum number of orbitals (" +
                                   std::to_string(nbits_half) + ").");
        }
        int k = 0;
        for (int p = 0; p < norb; ++p) {
            if (is_socc(p)) {
                socc_vec[k] = p;
                k++;
            }
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

#endif // _configuration_h_
