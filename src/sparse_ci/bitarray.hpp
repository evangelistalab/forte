#ifndef _bitarray_hpp_
#define _bitarray_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>
#include <array>

#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

/**
 * @brief This class represents an array of N bits. The bits are stored in
 *        groups called "words". Each word contains 64 bits.
 */
template <size_t N> class BitArray {
  public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;

    /// the number of bits
    static constexpr size_t nbits = N;

    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    /// this tests that N is a multiple of 64
    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

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
        if (val)
            getword(pos) |= maskbit(pos);
        else
            getword(pos) &= ~maskbit(pos);
    }

    void set_word(size_t pos, word_t word) { words_[pos] = word; }

    size_t get_nbits() const { return nbits; }

    /// default constructor
    BitArray() {}

    /// set all bits (including unused) to zero
    void zero() {
        for (size_t n = 0; n < nwords_; n++) {
            this->words_[n] = word_t(0);
        }
    }

    /// flip all bits
    void flip() {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] = ~words_[n];
        }
    }

    /// equal operator
    bool operator==(const BitArray<N>& lhs) const {
        for (size_t n = 0; n < nwords_; ++n) {
            if (this->words_[n] != lhs.words_[n])
                return false;
        }
        return true;
        //        // a possibly faster version without if
        //        bool val(true);
        //        for (size_t n = 0, bool val = true; n < nwords_; ++n) {
        //            val = val && (this->words_[n] != lhs.words_[n]);
        //        }
        //        return val;
    }

    /// Less than operator
    // TODO PERF: speedup with templated loop unrolling or avoid if
    bool operator<(const BitArray<N>& lhs) const {
        for (size_t n = nwords_; n > 1;) {
            --n;
            if (this->words_[n] > lhs.words_[n])
                return false;
            if (this->words_[n] < lhs.words_[n])
                return true;
        }
        return this->words_[0] < lhs.words_[0];
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

    /// Count the number of bits set to true in the words included in the range [begin,end)
    int count(size_t begin = 0, size_t end = nwords_) const {
        int c = 0;
        for (; begin < end; ++begin) {
            c += ui64_bit_count(this->words_[begin]);
        }
        return c;
    }

    /// Find the first bit set to one
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t find_first_one() const {
        for (size_t n = 0; n < nwords_; n++) {
            // find the first word != 0
            if (words_[n] != uint64_t(0)) {
                return ui64_find_lowest_one_bit(words_[n]);
            }
        }
        return ~uint64_t(0);
    }

    /// Clear the first bit set to one
    void clear_first_one() {
        for (size_t n = 0; n < nwords_; n++) {
            // find the first word != 0
            if (words_[n] != uint64_t(0)) {
                words_[n] = ui64_clear_lowest_one_bit(words_[n]);
                return;
            }
        }
    }

    /// Find the first bit set to one and clear it
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t find_and_clear_first_one() {
        for (size_t n = 0; n < nwords_; n++) {
            // find a word that is not 0
            if (words_[n] != uint64_t(0)) {
                // get the lowest set bit
                return ui64_find_and_clear_lowest_one_bit(words_[n]);
            }
        }
        // if the BitArray object is zero then return ~0
        return ~uint64_t(0);
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

    /// the index of the word where the bit in position pos is found
    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

    /// the word where bit in position pos is found
    word_t& getword(size_t pos) { return words_[whichword(pos)]; }
    const word_t& getword(size_t pos) const { return words_[whichword(pos)]; }

    /// the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

    /// a mask for bit pos in its corresponding word
    static constexpr word_t maskbit(size_t pos) noexcept {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    // ==> Private Data <==

/// The bits stored as a vector of words
#if PERFORMANCE_OPTIMIZATION
    word_t words_[nwords_];
#else
    //    word_t words_[nwords_] = {}; // all bits are set to zero
    std::array<word_t, nwords_> words_ = {};
#endif
};

} // namespace forte

#endif // _bitarray_hpp_
