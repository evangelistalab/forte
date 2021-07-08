/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _bitarray_hpp_
#define _bitarray_hpp_

#include <array>

#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

/**
 * @brief This class represents an array of N bits. The bits are stored in
 *        groups called "words". Each word contains 64 bits stored as
 *        64-bit unsigned integers.
 *
 *        Words are store in a std::array object:
 *          std::array<word_t, nwords_> words_ = {};
 *
 *        BirArray = |--64 bits--| |--64 bits--| |--64 bits--| ...
 *                       word[0]       word[1]       word[2]
 */
template <size_t N> class BitArray {
  public:
    /// alias for the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;

    /// the total number of bits (must be a multiple of 64)
    static constexpr size_t nbits = N;

    /// the number of bits in one word (64)
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    /// this tests that a word has 64 bits
    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    /// this tests that N is a multiple of 64
    static_assert(N % bits_per_word == 0, "The size of the BitArray (N) must be a multiple of 64");

    /// the number of words needed to store n bits
    static constexpr size_t bits_to_words(size_t n) {
        return n / bits_per_word + (n % bits_per_word != 0);
    }

    /// the number of words used to store the bits
    static constexpr size_t nwords_ = bits_to_words(nbits);

    // get the value of bit in position pos
    bool get_bit(size_t pos) const { return this->getword(pos) & maskbit(pos); }

    /// set bit in position pos to the value val
    void set_bit(size_t pos, bool val) {
        getword(pos) ^= (-val ^ getword(pos)) & maskbit(pos); // if-free implementation
    }

    /// set a word in position pos
    void set_word(size_t pos, word_t word) { words_[pos] = word; }

    /// return the number of bits
    size_t get_nbits() const { return nbits; }

    /// set all bits (including unused) to zero
    void zero() {
        if constexpr (N == 64) {
            words_[0] = word_t(0);
        } else if constexpr (N == 128) {
            words_[0] = word_t(0);
            words_[1] = word_t(0);
        } else {
            words_.fill(word_t(0));
        }
    }

    /// flip all bits
    void flip() {
        for (word_t& w : words_)
            w = ~w;
    }

    /// equal operator
    bool operator==(const BitArray<N>& lhs) const {
        if constexpr (N == 64) {
            return (this->words_[0] == lhs.words_[0]);
        } else if constexpr (N == 128) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]));
        } else if constexpr (N == 192) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]) and
                    (this->words_[2] == lhs.words_[2]));
        } else if constexpr (N == 256) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]) and
                    (this->words_[2] == lhs.words_[2]) and (this->words_[3] == lhs.words_[3]));
        } else {
            for (size_t n = 0; n < nwords_; ++n) {
                if (this->words_[n] != lhs.words_[n])
                    return false;
            }
            return true;
        }
        //        // a possibly faster version without if
        //        bool val(true);
        //        for (size_t n = 0, bool val = true; n < nwords_; ++n) {
        //            val = val && (this->words_[n] != lhs.words_[n]);
        //        }
        //        return val;
    }

    /// not operator
    BitArray<N> operator~() {
        BitArray<N> res(*this);
        res.flip();
        return res;
    }

    /// not equal operator
    bool operator!=(const BitArray<N>& lhs) const { return not(*this == lhs); }

    /// Less than operator
    bool operator<(const BitArray<N>& lhs) const {
        if constexpr (N == 64) {
            return (this->words_[0] < lhs.words_[0]);
        } else if constexpr (N == 128) {
            //  W1  W0  <
            //  >   >   F
            //  >   =   F
            //  >   <   F
            //  =   >   F
            //  =   =   F
            //  <   =   T
            //  <   >   T
            //  <   <   T
            //  =   <   T
            return (this->words_[1] < lhs.words_[1]) or
                   ((this->words_[1] == lhs.words_[1]) and (this->words_[0] < lhs.words_[0]));
        } else {
            for (size_t n = nwords_; n > 1;) {
                --n;
                if (this->words_[n] > lhs.words_[n])
                    return false;
                if (this->words_[n] < lhs.words_[n])
                    return true;
            }
            return this->words_[0] < lhs.words_[0];
        }
    }

    /// Bitwise OR operator (|)
    BitArray<N> operator|(const BitArray<N>& lhs) const {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] | lhs.words_[n];
        }
        return result;
    }

    /// Bitwise OR operator (|=)
    BitArray<N> operator|=(const BitArray<N>& lhs) const {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] |= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise XOR operator (^)
    BitArray<N> operator^(const BitArray<N>& lhs) const {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] ^ lhs.words_[n];
        }
        return result;
    }

    /// Bitwise XOR operator (^=)
    BitArray<N> operator^=(const BitArray<N>& lhs) {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] ^= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise AND operator (&)
    BitArray<N> operator&(const BitArray<N>& lhs) const {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] & lhs.words_[n];
        }
        return result;
    }

    /// Bitwise AND operator (&=)
    BitArray<N> operator&=(const BitArray<N>& lhs) {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] &= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise difference operator (-)
    BitArray<N> operator-(const BitArray<N>& lhs) const {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] & (~lhs.words_[n]);
        }
        return result;
    }

    /// Bitwise difference operator (-=)
    BitArray<N> operator-=(const BitArray<N>& lhs) const {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] &= ~lhs.words_[n];
        }
        return *this;
    }

    /// Count the number of bits set to true in the words included in the range [begin,end)
    int count(size_t begin = 0, size_t end = nwords_) const {
        int c = 0;
        for (; begin < end; ++begin) {
            c += ui64_bit_count(this->words_[begin]);
        }
        return c;
    }

    /// Find the first bit set to one (starting from the lowest index)
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t find_first_one() const {
        for (size_t n = 0; n < nwords_; n++) {
            // find the first word != 0
            if (words_[n] != word_t(0)) {
                return ui64_find_lowest_one_bit(words_[n]) + n * bits_per_word;
            }
        }
        return ~word_t(0);
    }

    /// Clear the first bit set to one (starting from the lowest index)
    void clear_first_one() {
        for (size_t n = 0; n < nwords_; n++) {
            // find the first word != 0
            if (words_[n] != word_t(0)) {
                words_[n] = ui64_clear_lowest_one_bit(words_[n]);
                return;
            }
        }
    }

    /// Find the first bit set to one and clear it (starting from the lowest index)
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t find_and_clear_first_one() {
        for (size_t n = 0; n < nwords_; n++) {
            // find a word that is not 0
            if (words_[n] != word_t(0)) {
                // get the lowest set bit
                return ui64_find_and_clear_lowest_one_bit(words_[n]) + n * bits_per_word;
            }
        }
        // if the BitArray object is zero then return ~0
        return ~word_t(0);
    }

    /// Find the first bit set to one and clear it (starting from the lowest index)
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t fast_find_and_clear_first_one(size_t n) {
        n = whichword(n);
        for (; n < nwords_; n++) {
            // find a word that is not 0
            if (words_[n] != word_t(0)) {
                // get the lowest set bit
                return ui64_find_and_clear_lowest_one_bit(words_[n]) + n * bits_per_word;
            }
        }
        // if the BitArray object is zero then return ~0
        return ~word_t(0);
    }

    /// Implements the operation: (a & b) == b
    bool fast_a_and_b_equal_b(const BitArray<N>& b) const {
        bool result = false;
        for (size_t n = 0; n < nwords_; n++) {
            result += ((words_[n] & b.words_[n]) != b.words_[n]);
        }
        return not result;
    }

    /// Implements the operation: a - b == 0
    bool fast_a_minus_b_eq_zero(const BitArray<N>& b) const {
        bool result = false;
        for (size_t n = 0; n < nwords_; n++) {
            result += words_[n] & (~b.words_[n]);
        }
        return not result;
    }

    /// Implements the operation: a & b == 0
    bool fast_a_and_b_eq_zero(const BitArray<N>& b) const {
        bool result = false;
        for (size_t n = 0; n < nwords_; n++) {
            result += words_[n] & b.words_[n];
        }
        return not result;
    }

    /// Implements the operation: count(a ^ b)
    int fast_a_xor_b_count(const BitArray<N>& b) const {
        if constexpr (N == 64) {
            return ui64_bit_count(words_[0] ^ b.words_[0]);
        } else if constexpr (N == 128) {
            return ui64_bit_count(words_[0] ^ b.words_[0]) +
                   ui64_bit_count(words_[1] ^ b.words_[1]);
        } else if constexpr (N == 192) {
            return ui64_bit_count(words_[0] ^ b.words_[0]) +
                   ui64_bit_count(words_[1] ^ b.words_[1]) +
                   ui64_bit_count(words_[2] ^ b.words_[2]);
        } else if constexpr (N == 256) {
            return ui64_bit_count(words_[0] ^ b.words_[0]) +
                   ui64_bit_count(words_[1] ^ b.words_[1]) +
                   ui64_bit_count(words_[2] ^ b.words_[2]) +
                   ui64_bit_count(words_[3] ^ b.words_[3]);
        } else {
            int c = 0;
            for (size_t n = 0; n < nwords_; n++) {
                c += ui64_bit_count(words_[n] ^ b.words_[n]);
            }
            return c;
        }
    }

    /// Return the sign of a_n applied to this determinant
    /// This function ignores if bit n is set or not
    double slater_sign(int n) const {
        if constexpr (N == 64) {
            return ui64_sign(words_[0], n);
        } else {
            size_t count = 0;
            // count all the preceeding bits only if we are looking past the first word
            if (static_cast<size_t>(n) >= bits_per_word) {
                size_t last_full_word = whichword(n);
                for (size_t k = 0; k < last_full_word; ++k) {
                    count += ui64_bit_count(words_[k]);
                }
            }
            return (count % 2 == 0) ? ui64_sign(getword(n), whichbit(n))
                                    : -ui64_sign(getword(n), whichbit(n));
        }
    }

    /// Return the sign for a pair of second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    /// There are no restrictions on n and m
    double slater_sign(int n, int m) const {
        if constexpr (N == 64) {
            return ui64_sign(words_[0], n, m);
        } else if constexpr (N == 128) {
            // XXXXXXXX YYYYYYYY
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccc
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccccc
            // cccccc
            // XmXXXXXX YYYYYnYY (case 2)
            //   cccccc ccccc
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            return ui64_sign_reverse(words_[word_m], whichbit(m)) *
                   ui64_sign(words_[word_n], whichbit(n));
        } else {
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            size_t count = 0;
            // count the number of bits in bitween the words of m and n
            for (size_t k = word_m + 1; k < word_n; ++k) {
                count += ui64_bit_count(words_[k]);
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            double sign = ui64_sign_reverse(words_[word_m], whichbit(m)) *
                          ui64_sign(words_[word_n], whichbit(n));
            return (count % 2 == 0) ? sign : -sign;
        }
    }

    /// Return the sign of a_n applied to this determinant
    /// this version is inefficient and should be used only for testing/debugging
    double slater_sign_safe(int n) const {
        size_t count = 0;
        for (int k = 0; k < n; ++k) {
            if (get_bit(k))
                count++;
        }
        return (count % 2 == 0) ? 1.0 : -1.0;
    }

    /// Returns a hash value for a BitArray object
    struct Hash {
        std::size_t operator()(const BitArray<N>& d) const {
            if constexpr (N == 64) {
                return d.words_[0];
            } else if constexpr (N == 128) {
                return ((d.words_[0] * 13466917) + d.words_[1]) % 1405695061;
            } else {
                std::uint64_t seed = nwords_;
                for (auto& w : d.words_) {
                    hash_combine_uint64(seed, w);
                }
                return seed;
            }
        }
    };

  protected:
    // ==> Private Functions <==

    // These functions are used to address bits in the BitArray.
    // They should not be used outside the class because they contain details of the implementation.

    /// the index of the word where the bit in position pos is found
    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

    /// the word where bit in position pos is found
    word_t& getword(size_t pos) { return words_[whichword(pos)]; }

    /// the word where bit in position pos is found (const version)
    const word_t& getword(size_t pos) const { return words_[whichword(pos)]; }

    /// the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

    /// a mask for bit pos in its corresponding word
    static constexpr word_t maskbit(size_t pos) noexcept {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    // ==> Private Data <==

#if PERFORMANCE_OPTIMIZATION
    /// The bits stored as a vector of words (uninitialized)
    std::array<word_t, nwords_> words_;
#else
    /// The bits stored as a vector of words (initialized to zero at construction)
    std::array<word_t, nwords_> words_ = {};
#endif
};

} // namespace forte

#endif // _bitarray_hpp_
