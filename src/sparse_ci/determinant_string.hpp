#ifndef _determinant_string_hpp_
#define _determinant_string_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

template <size_t N> class StringImpl {
  public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;
    /// the number of bits
    static constexpr size_t nbits = N;
    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    static_assert(N % (64) == 0,
                  "The number of bits in the Determinant class must be a multiple of 64");

    /// the number of words needed to store n bits
    static constexpr size_t nwords(size_t n) {
        return n / bits_per_word + (n % bits_per_word == 0 ? 0 : 1);
    }
    /// the number of words used to store the bits
    static constexpr size_t nwords_ = nwords(nbits);

    StringImpl() {}

    bool get_bit(size_t pos) const { return this->getword(pos) & maskbit(pos); }

    void set_word(size_t pos, word_t word) { words_[pos] = word; }

    /// Return the sign of a_n applied to this determinant
    /// This function ignores if bit n is set or not
    double slater_sign(int n) const {
        // with constexpr we compile only one of these cases
        size_t count = 0;
        // count all the preceeding bits only if we are looking past the first word
        if (n >= bits_per_word) {
            size_t last_full_word = whichword(n);
            for (size_t k = 0; k < last_full_word; ++k) {
                count += ui64_bit_count(words_[k]);
            }
        }
        return (count % 2 == 0) ? ui64_sign(getword(n), whichbit(n))
                                : -ui64_sign(getword(n), whichbit(n));
    }

    /// Return the sign for a pair of second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    double slater_sign(int n, int m) const {
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
        double sign =
            ui64_sign_reverse(words_[word_m], whichbit(m)) * ui64_sign(words_[word_n], whichbit(n));
        return (count % 2 == 0) ? sign : -sign;
    }

    int count() const {
        int c = 0;
        for (auto const w : words_) {
            c += ui64_bit_count(w);
        }
        return c;
    }

    /// Comparison operator
    bool operator==(const StringImpl<N>& lhs) const {
        for (size_t n = 0; n < nwords_; n++) {
            if (this->words_[n] != lhs.words_[n])
                return false;
        }
        return true;
    }

    /// Less than operator
    bool operator<(const StringImpl<N>& lhs) const {
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
    StringImpl<N> operator|(const StringImpl<N>& lhs) const {
        StringImpl<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = this->words_[n] | lhs.words_[n];
        }
        return result;
    }

    /// Bitwise XOR operator (^)
    StringImpl<N> operator^(const StringImpl<N>& lhs) const {
        StringImpl<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = this->words_[n] ^ lhs.words_[n];
        }
        return result;
    }

    /// Bitwise XOR operator (^)
    StringImpl<N> operator^=(const StringImpl<N>& lhs) {
        for (size_t n = 0; n < nwords_; n++) {
            this->words_[n] ^= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise AND operator (&)
    StringImpl<N> operator&(const StringImpl<N>& lhs) const {
        StringImpl<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = this->words_[n] & lhs.words_[n];
        }
        return result;
    }

    /// Bitwise AND operator (&)
    StringImpl<N> operator&=(const StringImpl<N>& lhs) {
        StringImpl<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            this->words_[n] &= lhs.words_[n];
        }
        return result;
    }

    uint64_t lowest_one_index() const {
        for (size_t n = 0; n < nwords_; n++) {
            uint64_t idx = lowest_one_idx(words_[n]);
            if (idx != ~uint64_t(0)) {
                return idx;
            }
        }
        return ~uint64_t(0);
    }

    void clear_lowest_one() {
        for (size_t n = 0; n < nwords_; n++) {
            if (this->words_[n] != uint64_t(0)) {
                this->words_[n] = clear_lowest_one_bit(this->words_[n]);
                return;
            }
        }
    }

    /// Hash function
    struct Hash {
        std::size_t operator()(const StringImpl<N>& s) const {
            if constexpr (N == 128) {
                return s.words_[0];
            } else {
                std::uint64_t seed = 0;
                for (auto& w : s.words_) {
                    hash_combine_uint64(seed, w);
                }
                return seed;
            }
        }
    };

  private:
    // ==> Private Functions <==

    /// the index of the word where bit pos is found
    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

    /// the word where bit pos is found
    word_t& getword(size_t pos) { return words_[whichword(pos)]; }
    const word_t& getword(size_t pos) const { return words_[whichword(pos)]; }

    /// the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

    /// a mask for bit pos in its corresponding word
    static constexpr word_t maskbit(size_t pos) {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    word_t words_[nwords_];
};

} // namespace forte

#endif // _determinant_hpp_
