#ifndef _bitarray_hpp_
#define _bitarray_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

template <size_t N> class BitArray {
  public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;

    /// the number of bits
    static constexpr size_t nbits = N;

    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);
    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    /// the number of words needed to store n bits
    static constexpr size_t bits_to_words(size_t n) {
        return n / bits_per_word + (n % bits_per_word == 0 ? 0 : 1);
    }

    /// the number of words used to store the bits
    static constexpr size_t nwords_ = bits_to_words(nbits);

    // get the value of bit pos
    bool get_bit(size_t pos) const { return this->getword(pos) & maskbit(pos); }

    size_t get_nbits() const { return nbits; }

    /// Default constructor
    BitArray() {}

    /// Set all bits (including unused) to zero
    void zero() {
        for (size_t n = 0; n < nwords_; n++) {
            this->words_[n] = 0;
        }
    }

    /// Flip all bits and return a reference to this object
    void flip() {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] = ~words_[n];
        }
    }

    /// Equal operator
    bool operator==(const BitArray<N>& lhs) const {
        for (size_t n = 0; n < nwords_; ++n) {
            if (this->words_[n] != lhs.words_[n])
                return false;
        }
        return true;
    }

    /// Less than operator
    // TODO PERF: speedup with templated loop unrolling
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

    /// Count the number of bits set to true from the begin position up to the end (excluded)
    int count(size_t begin = 0, size_t end = nwords_) const {
        int c = 0;
        for (; begin < end; ++begin) {
            c += ui64_bit_count(this->words_[begin]);
        }
        return c;
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

    // ==> Private Data <==

/// The bits stored as a vector of words
#if PERFORMANCE_OPTIMIZATION
    word_t words_[nwords_];
#else
    word_t words_[nwords_] = {}; // all bits are set to zero
#endif
};

} // namespace forte

#endif // _bitarray_hpp_
