//#ifndef _determinant_string_hpp_
//#define _determinant_string_hpp_

//#include "bitarray.hpp"
//#include "bitwise_operations.hpp"

//namespace forte {

//#define PERFORMANCE_OPTIMIZATION 0

//template <size_t N> class StringImpl : public BitArray<N> {
//  public:
//    // Since the template parent (BitArray) of this template class is not instantiated during the
//    // compilation pass, here we declare all the member variables and functions inherited and used
//    //    using BitArray<N>::nbits;
//    //    using BitArray<N>::bits_per_word;
//    //    using BitArray<N>::nwords_;
//    //    using BitArray<N>::words_;
//    using BitArray<N>::words_;
//    using BitArray<N>::set_word;
//    using BitArray<N>::operator|;
//    using BitArray<N>::operator^;
//    using BitArray<N>::operator&;

//    //    using BitArray<N>::count;
//    //    using BitArray<N>::get_bit;
//    //    using BitArray<N>::set_bit;
//    //    using BitArray<N>::maskbit;
//    //    using BitArray<N>::whichbit;
//    //    using BitArray<N>::whichword;
//    //    using BitArray<N>::getword;

//    //    /// the type used to represent a word (a 64 bit unsigned integer)
//    //    using word_t = uint64_t;
//    //    /// the number of bits
//    //    static constexpr size_t nbits = N;
//    //    /// the number of bits in one word
//    //    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

//    //    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

//    //    static_assert(N % (64) == 0,
//    //                  "The number of bits in the Determinant class must be a multiple of 64");

//    //    /// the number of words needed to store n bits
//    //    static constexpr size_t nwords(size_t n) {
//    //        return n / bits_per_word + (n % bits_per_word == 0 ? 0 : 1);
//    //    }
//    //    /// the number of words used to store the bits
//    //    static constexpr size_t nwords_ = nwords(nbits);

//    StringImpl() : BitArray<N>() {}
//    StringImpl(const BitArray<N>& ba) : BitArray<N>(ba) {}

//    //    /// Comparison operator
//    //    bool operator==(const StringImpl<N>& lhs) const {
//    //        for (size_t n = 0; n < nwords_; n++) {
//    //            if (words_[n] != lhs.words_[n])
//    //                return false;
//    //        }
//    //        return true;
//    //    }

//    //    /// Less than operator
//    //    bool operator<(const StringImpl<N>& lhs) const {
//    //        for (size_t n = nwords_; n > 1;) {
//    //            --n;
//    //            if (words_[n] > lhs.words_[n])
//    //                return false;
//    //            if (words_[n] < lhs.words_[n])
//    //                return true;
//    //        }
//    //        return words_[0] < lhs.words_[0];
//    //    }

//    //    /// Bitwise OR operator (|)
//    //    StringImpl<N> operator|(const StringImpl<N>& lhs) const { return *this | lhs; }
//    //    /// Bitwise XOR operator (^)
//    //    StringImpl<N> operator^(const StringImpl<N>& lhs) const { return *this ^ lhs; }
//    //    /// Bitwise AND operator (&)
//    //    StringImpl<N> operator&(const StringImpl<N>& lhs) const { return *this & lhs; }

//    //    /// Bitwise OR operator (|=)
//    //    BitArray<N> operator|=(const BitArray<N>& lhs) const {
//    //        for (size_t n = 0; n < nwords_; n++) {
//    //            words_[n] |= lhs.words_[n];
//    //        }
//    //        return *this;
//    //    }

//    //    /// Bitwise XOR operator (^=)
//    //    BitArray<N> operator^=(const BitArray<N>& lhs) {
//    //        for (size_t n = 0; n < nwords_; n++) {
//    //            words_[n] ^= lhs.words_[n];
//    //        }
//    //        return *this;
//    //    }

//    //    /// Bitwise AND operator (&)
//    //    BitArray<N> operator&(const BitArray<N>& lhs) const {
//    //        BitArray<N> result;
//    //        for (size_t n = 0; n < nwords_; n++) {
//    //            result.words_[n] = words_[n] & lhs.words_[n];
//    //        }
//    //        return result;
//    //    }

//    //    /// Bitwise AND operator (&=)
//    //    BitArray<N> operator&=(const BitArray<N>& lhs) {
//    //        for (size_t n = 0; n < nwords_; n++) {
//    //            words_[n] &= lhs.words_[n];
//    //        }
//    //        return *this;
//    //    }

//    /// Hash function
//    struct Hash {
//        std::size_t operator()(const StringImpl<N>& s) const {
//            if constexpr (N == 128) {
//                return s.words_[0];
//            } else {
//                std::uint64_t seed = 0;
//                for (auto& w : s.words_) {
//                    hash_combine_uint64(seed, w);
//                }
//                return seed;
//            }
//        }
//    };

//  private:
//    // ==> Private Functions <==

//    //    /// the index of the word where bit pos is found
//    //    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

//    //    /// the word where bit pos is found
//    //    word_t& getword(size_t pos) { return words_[whichword(pos)]; }
//    //    const word_t& getword(size_t pos) const { return words_[whichword(pos)]; }

//    //    /// the index of a bit within a word
//    //    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

//    //    /// a mask for bit pos in its corresponding word
//    //    static constexpr word_t maskbit(size_t pos) {
//    //        return (static_cast<word_t>(1)) << whichbit(pos);
//    //    }

//    //    word_t words_[nwords_];
//};

//} // namespace forte

//#endif // _determinant_hpp_
