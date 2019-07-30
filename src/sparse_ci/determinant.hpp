#ifndef _determinant_hpp_
#define _determinant_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

#include "bitwise_operations.hpp"

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

enum class DetSpinType { Alpha, Beta };

template <size_t N> class StringImpl {
  public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;
    /// the number of bits
    static constexpr size_t nbits = N;
    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    static_assert(N % (128) == 0,
                  "The number of bits in the Determinant class must be a multiple of 128");

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
template <size_t N> class DeterminantImpl {
  public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;
    /// the number of bits
    static constexpr size_t nbits = N;
    /// the number of bits divided by two
    static constexpr size_t nbits_half = nbits / 2;
    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    static_assert(N % (128) == 0,
                  "The number of bits in the Determinant class must be a multiple of 128");

    /// the number of words needed to store n bits
    static constexpr size_t nwords(size_t n) {
        return n / bits_per_word + (n % bits_per_word == 0 ? 0 : 1);
    }
    /// the number of words used to store the bits
    static constexpr size_t nwords_ = 2 * nwords(nbits_half);

    /// half of the words used to store the bits
    static constexpr size_t nwords_half = nwords(nbits_half);

    /// the starting bit for beta orbitals
    static constexpr size_t beta_bit_offset = nwords_half * bits_per_word;

    /// Default constructor
    DeterminantImpl() {}

    //    explicit DeterminantImpl(const std::vector<bool>& occupation);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit DeterminantImpl(const std::vector<bool>& occupation_a,
                             const std::vector<bool>& occupation_b) {
        int size = occupation_a.size();
        for (int p = 0; p < size; ++p) {
            this->set_alfa_bit(p, occupation_a[p]);
            this->set_beta_bit(p, occupation_b[p]);
        }
    }

    /// String constructor. Convert a std::string to a determinant.
    /// E.g. DeterminantImpl<64>("0011") gives the determinant|0011>
    DeterminantImpl(const std::string& str) { set_str(*this, str); }

    // Functions to access bits

    /// get the value of bit pos
    bool get_bit(size_t pos) const { return this->getword(pos) & maskbit(pos); }
    /// get the value of alfa bit pos
    bool get_alfa_bit(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[0] & maskbit(pos);
        } else {
            return this->getword(pos) & maskbit(pos);
        }
    }
    /// get the value of beta bit pos
    bool get_beta_bit(size_t pos) const {
        if constexpr (nbits == 128) {
            return words_[1] & maskbit(pos);
        } else {
            return this->getword(pos + beta_bit_offset) & maskbit(pos);
        }
    }

    // Functions to set bits

    /// set bit pos to val
    DeterminantImpl<nbits>& set_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos) |= maskbit(pos);
        else
            this->getword(pos) &= ~maskbit(pos);
        return *this;
    }

    /// set alfa bit pos to val
    DeterminantImpl<nbits>& set_alfa_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos) |= maskbit(pos);
        else
            this->getword(pos) &= ~maskbit(pos);
        return *this;
    }
    /// set beta bit pos to val
    DeterminantImpl<nbits>& set_beta_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos + beta_bit_offset) |= maskbit(pos);
        else
            this->getword(pos + beta_bit_offset) &= ~maskbit(pos);
        return *this;
    }

    /// Set all bits (including unused) to zero
    void zero() {
        for (size_t n = 0; n < nwords_; n++) {
            this->words_[n] = 0;
        }
    }

    // Comparison operators

    /// Equal operator
    // TODO PERF: speedup with templated loop unrolling
    bool operator==(const DeterminantImpl<N>& lhs) const {
        for (size_t n = 0; n < nwords_; n++) {
            if (this->words_[n] != lhs.words_[n])
                return false;
        }
        return true;
    }

    /// Less than operator
    // TODO PERF: speedup with templated loop unrolling
    bool operator<(const DeterminantImpl<N>& lhs) const {
        for (size_t n = nwords_; n > 1;) {
            --n;
            if (this->words_[n] > lhs.words_[n])
                return false;
            if (this->words_[n] < lhs.words_[n])
                return true;
        }
        return this->words_[0] < lhs.words_[0];
    }

    static bool less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
        for (size_t n = nwords_; n > 1;) {
            --n;
            if (rhs.words_[n] > lhs.words_[n])
                return false;
            if (rhs.words_[n] < lhs.words_[n])
                return true;
        }
        return rhs.words_[0] < lhs.words_[0];
    }

    static bool reverse_less_than(const DeterminantImpl<N>& rhs, const DeterminantImpl<N>& lhs) {
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

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const {
        std::vector<int> occ;
        for (int p = 0; p < norb; ++p) {
            if (this->get_alfa_bit(p)) {
                occ.push_back(p);
            }
        }
        return occ;
    }

    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ(int norb) const {
        std::vector<int> occ;
        for (int p = 0; p < norb; ++p) {
            if (this->get_beta_bit(p)) {
                occ.push_back(p);
            }
        }
        return occ;
    }

    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir(int norb) const {
        std::vector<int> vir;
        for (int p = 0; p < norb; ++p) {
            if (not this->get_alfa_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const {
        std::vector<int> vir;
        for (int p = 0; p < norb; ++p) {
            if (not this->get_beta_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Apply the alpha creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_alfa_bit(int n) {
        if (this->get_alfa_bit(n))
            return 0.0;
        this->set_alfa_bit(n, true);
        return this->slater_sign_a(n);
    }

    /// Apply the beta creation operator a^+_n to this determinant
    /// If orbital n is unoccupied, create the electron and return the sign
    /// If orbital n is occupied, do not modify the determinant and return 0
    double create_beta_bit(int n) {
        if (this->get_beta_bit(n))
            return 0.0;
        this->set_beta_bit(n, true);
        return this->slater_sign_b(n);
    }

    /// Apply the alpha annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_alfa_bit(int n) {
        if (not this->get_alfa_bit(n))
            return 0.0;
        this->set_alfa_bit(n, false);
        return this->slater_sign_a(n);
    }

    /// Apply the beta annihilation operator a^+_n to this determinant
    /// If orbital n is occupied, annihilate the electron and return the sign
    /// If orbital n is unoccupied, do not modify the determinant and return 0
    double destroy_beta_bit(int n) {
        if (not this->get_beta_bit(n))
            return 0.0;
        this->set_beta_bit(n, false);
        return this->slater_sign_b(n);
    }

    // THE FOLLOWING FUNCTIONS ARE NOT TESTED

    /// Return the dimensions for occupied alpha orbitals
    std::vector<std::vector<int>> get_asym_occ(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for occupied beta orbitals
    std::vector<std::vector<int>> get_bsym_occ(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for virtual alpha orbitals
    std::vector<std::vector<int>> get_asym_vir(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for virtual beta orbitals
    std::vector<std::vector<int>> get_bsym_vir(int norb, std::vector<int> act_mo) const;

    /// Return the sign of a_n applied to this determinant
    /// this version is inefficient and should be used only for testing/debugging
    double slater_sign_safe(int n) const {
        size_t count = 0;
        for (size_t k = 0; k < n; ++k) {
            if (this->get_bit(k))
                count++;
        }
        return (count % 2 == 0) ? 1.0 : -1.0;
    }

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
            return ui64_sign(words_[1], n) * (ui64_bit_count(words_[0]) % 2 == 0 ? 1.0 : -1.0);
        } else {
            return slater_sign(n + beta_bit_offset);
        }
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

    /// Count the number of bits set to true
    int count() const {
        int c = 0;
        for (auto const w : words_) {
            c += ui64_bit_count(w);
        }
        return c;
    }

    int count_alfa() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return ui64_bit_count(words_[0]);
        } else if (N == 256) {
            return ui64_bit_count(words_[0]) + ui64_bit_count(words_[1]);
        } else {
            int na = 0;
            // count the number of bits in the alpha words
            for (int k = 0; k < nwords_half; ++k) {
                na += ui64_bit_count(words_[k]);
            }
            return na;
        }
    }

    /// Count the number of beta bits set to true
    int count_beta() const {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return ui64_bit_count(words_[1]);
        } else if (N == 256) {
            return ui64_bit_count(words_[2]) + ui64_bit_count(words_[3]);
        } else {
            int nb = 0;
            // count the number of bits in the beta words
            for (int k = nwords_half; k < nwords_; ++k) {
                nb += ui64_bit_count(words_[k]);
            }
            return nb;
        }
    };

    /// Return the number of alpha/beta pairs
    int npair() const {
        int count = 0;
        for (int k = 0; k < nwords_half; ++k) {
            count += ui64_bit_count(words_[k] & words_[k + nwords_half]);
        }
        return count;
    }

    /// Flip all bits and return a reference to this object
    DeterminantImpl<N>& flip() {
        for (size_t k = 0; k < nwords_; ++k) {
            ~words_[k];
        }
        return *this;
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

    /// Save the Slater determinant as a string
    /// @param n number of bits to print (number of MOs)
    //    std::string str(int n = nbits_half) const { return str(*this, n); }

    /// Hash function
    struct Hash {
        std::size_t operator()(const DeterminantImpl<N>& d) const {
            if constexpr (N == 128) {
                return ((d.words_[0] * 13466917) + d.words_[1]) % 1405695061;
            } else {
                std::uint64_t seed = 0;
                for (auto& w : d.words_) {
                    hash_combine_uint64(seed, w);
                }
                return seed;
            }
        }
    };

    StringImpl<nbits_half> get_alfa_bits() const {
        StringImpl<nbits_half> s;
        for (int i = 0; i < nwords_half; i++) {
            s.set_word(i, words_[i]);
        }
        return s;
    }

    StringImpl<nbits_half> get_beta_bits() const {
        StringImpl<nbits_half> s;
        for (int i = 0; i < nwords_half; i++) {
            s.set_word(i, words_[nwords_half + i]);
        }
        return s;
    }

    StringImpl<nbits_half> get_bits(DetSpinType spin_type) {
        return (spin_type == DetSpinType::Alpha ? this->get_alfa_bits() : this->get_beta_bits());
    }

    /// Zero the alpha or beta part of a determinant
    [[deprecated(
        "The zero_spin function should be replaced with methods from the String class")]] void
    zero_spin(DetSpinType spin_type) {
        if (spin_type == DetSpinType::Alpha) {
            for (size_t n = 0; n < nwords_half; n++)
                words_[n] = u_int64_t(0);
        } else {
            for (size_t n = nwords_half; n < nwords_; n++)
                words_[n] = u_int64_t(0);
        }
    }

    // END OF FUNCTIONS NOT TESTED

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

    /// a function to accumulate hash values of 64 bit unsigned integers
    /// based on boost/functional/hash/hash.hpp
    static inline void hash_combine_uint64(uint64_t& seed, uint64_t value) {
        const uint64_t m = 0xc6a4a7935bd1e995ULL;
        const int r = 47;
        value *= m;
        value ^= value >> r;
        value *= m;
        seed ^= value;
        seed *= m;
        seed += 0xe6546b64;
        // alternative one-liner
        // seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

// ==> Private Data <==

/// The bits stored as a vector of words
#if PERFORMANCE_OPTIMIZATION
    word_t words_[nwords_];
#else
    word_t words_[nwords_] = {}; // all bits are set to zero
#endif
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
std::vector<std::vector<int>> get_asym_occ(const DeterminantImpl<N>& d, std::vector<int> act_mo) {

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
std::vector<std::vector<int>> get_bsym_occ(const DeterminantImpl<N>& d, std::vector<int> act_mo) {
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
std::vector<std::vector<int>> get_asym_vir(const DeterminantImpl<N>& d, std::vector<int> act_mo) {
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
std::vector<std::vector<int>> get_bsym_vir(const DeterminantImpl<N>& d, std::vector<int> act_mo) {
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

template <size_t N> double spin2(const DeterminantImpl<N>& lhs, const DeterminantImpl<N>& rhs) {
    int size = DeterminantImpl<N>::nbits_half;
    const DeterminantImpl<N>& I = lhs;
    const DeterminantImpl<N>& J = rhs;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    int na = 0;
    int nb = 0;
    int npair = 0;
    // Count how many differences in mos are there and the number of alpha/beta
    // electrons
    for (int n = 0; n < size; ++n) {
        if (I.get_alfa_bit(n) != J.get_alfa_bit(n))
            nadiff++;
        if (I.get_beta_bit(n) != J.get_beta_bit(n))
            nbdiff++;
        if (I.get_alfa_bit(n))
            na++;
        if (I.get_beta_bit(n))
            nb++;
        if ((I.get_alfa_bit(n) and I.get_beta_bit(n)))
            npair += 1;
    }
    nadiff /= 2;
    nbdiff /= 2;

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
        for (int p = 0; p < size; ++p) {
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

template <size_t N>
void enforce_spin_completeness(std::vector<DeterminantImpl<N>>& det_space, int nmo) {
    /*
        std::unordered_map<DeterminantImpl<N>, bool, DeterminantImpl<N>::Hash> det_map;
        // Add all determinants to the map, assume set is mostly spin complete
        for (auto& I : det_space) {
            det_map[I] = true;
        }
        // Loop over determinants
        size_t ndet_added = 0;
        std::vector<size_t> closed(nmo, 0);
        std::vector<size_t> open(nmo, 0);
        std::vector<size_t> open_bits(nmo, 0);
        for (size_t I = 0, det_size = det_space.size(); I < det_size; ++I) {
            const DeterminantImpl<N>& det = det_space[I];
            // outfile->Printf("\n  Original determinant: %s", det.str().c_str());
            for (int i = 0; i < nmo; ++i) {
                closed[i] = open[i] = 0;
                open_bits[i] = false;
            }
            int naopen = 0;
            int nbopen = 0;
            int nclosed = 0;
            for (int i = 0; i < nmo; ++i) {
                if (det.get_alfa_bit(i) and (not det.get_beta_bit(i))) {
                    open[naopen + nbopen] = i;
                    naopen += 1;
                } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)) {
                    open[naopen + nbopen] = i;
                    nbopen += 1;
                } else if (det.get_alfa_bit(i) and det.get_beta_bit(i)) {
                    closed[nclosed] = i;
                    nclosed += 1;
                }
            }

            if (naopen + nbopen == 0)
                continue;

            // Generate the strings 1111100000
            //                      {nao}{nbo}
            for (int i = 0; i < nbopen; ++i)
                open_bits[i] = false; // 0
            for (int i = nbopen; i < naopen + nbopen; ++i)
                open_bits[i] = true; // 1
            do {
                DeterminantImpl<N> new_det;
                for (int c = 0; c < nclosed; ++c) {
                    new_det.set_alfa_bit(closed[c], true);
                    new_det.set_beta_bit(closed[c], true);
                }
                for (int o = 0; o < naopen + nbopen; ++o) {
                    if (open_bits[o]) { //? not
                        new_det.set_alfa_bit(open[o], true);
                    } else {
                        new_det.set_beta_bit(open[o], true);
                    }
                }
                if (det_map.count(new_det) == 0) {
                    det_space.push_back(new_det);
                    det_map[new_det] = true;
                    // outfile->Printf("\n  added determinant:    %s", new_det.str().c_str());
                    ndet_added++;
                }
            } while (std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
        }
        // if( ndet_added > 0 ){
        //    outfile->Printf("\n\n  Determinant space is spin incomplete!");
        //    outfile->Printf("\n  %zu more determinants were needed.", ndet_added);
        //}else{
        //    outfile->Printf("\n\n  Determinant space is spin complete.");
        //}
        */ // TODO onedet: re-enable this function
}

} // namespace forte

#endif // _determinant_hpp_
