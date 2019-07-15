#ifndef _determinant_hpp_
#define _determinant_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

#include "bitwise_operations.hpp"

#define PERFORMANCE_OPTIMIZATION 0

/**
 * @brief A class to represent a Slater determinant with N spin orbitals
 *
 * The orbitals are assumed to be sorted as
 *  [word 1][word 2][word 3][word 4]...
 */
template <size_t N> class Determinant {
public:
    /// the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;
    /// the number of bits
    static constexpr size_t nbits = N;
    /// the number of bits divided by two
    static constexpr size_t nbits_half = nbits / 2;
    /// the number of bits in one word
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);
    /// the starting point for bits that represent beta orbitals
    static constexpr size_t beta_offset = nbits / 2;
    /// the number of words needed to store n bits
    static constexpr size_t nwords(size_t n) {
        return n / bits_per_word + (n % bits_per_word == 0 ? 0 : 1);
    }
    /// the number of words used to store the bits
    static constexpr size_t nwords_ = nwords(nbits);
    /// the number of bits used
    static constexpr size_t nbits_used_ = nwords_ * bits_per_word;

    /// Default constructor
    Determinant(){}

    /// String constructor. Convert a std::string to a determinant.
    /// E.g. Determinant<4>("0011") gives the determinant|0011>
    Determinant(const std::string &str) { set_str(*this, str); }

    // Functions to access bits

    /// get the value of bit pos
    bool get_bit(size_t pos) const {
        return this->getword(pos) & maskbit(pos);
    }
    /// get the value of alfa bit pos
    bool get_alfa_bit(size_t pos) const {
        return this->getword(pos) & maskbit(pos);
    }
    /// get the value of beta bit pos
    bool get_beta_bit(size_t pos) const {
        return this->getword(pos + beta_offset) & maskbit(pos + beta_offset);
    }

    // Functions to set bits

    /// set bit pos to val
    Determinant<nbits> &set_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos) |= maskbit(pos);
        else
            this->getword(pos) &= ~maskbit(pos);
        return *this;
    }

    /// set alfa bit pos to val
    Determinant<nbits> &set_alfa_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos) |= maskbit(pos);
        else
            this->getword(pos) &= ~maskbit(pos);
        return *this;
    }
    /// set beta bit pos to val
    Determinant<nbits> &set_beta_bit(size_t pos, bool val) {
        if (val)
            this->getword(pos + beta_offset) |= maskbit(pos + beta_offset);
        else
            this->getword(pos + beta_offset) &= ~maskbit(pos + beta_offset);
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
    bool operator==(const Determinant<N> &lhs) const {
        for (size_t n = 0; n < nwords_; n++) {
            if (this->words_[n] != lhs.words_[n])
                return false;
        }
        return true;
    }

    /// Less than operator
    // TODO PERF: speedup with templated loop unrolling
    bool operator<(const Determinant<N> &lhs) const {
        for (size_t n = nwords_; n > 0; ) {
            --n;
            if (this->words_[n] > lhs.words_[n])
                return false;
            if (this->words_[n] < lhs.words_[n])
                return true;
        }
        return false;
    }

    // THE FOLLOWING FUNCTIONS ARE NOT TESTED

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ(int norb) const{
        std::vector<int> occ;
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
        for (int p = 0; p < norb; ++p) {
            if (not get_alfa_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir(int norb) const{
        std::vector<int> vir;
        for (int p = 0; p < norb; ++p) {
            if (not get_beta_bit(p)) {
                vir.push_back(p);
            }
        }
        return vir;
    }

    /// Return the dimensions for occupied alpha orbitals
    std::vector<std::vector<int>> get_asym_occ(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for occupied beta orbitals
    std::vector<std::vector<int>> get_bsym_occ(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for virtual alpha orbitals
    std::vector<std::vector<int>> get_asym_vir(int norb, std::vector<int> act_mo) const;
    /// Return the dimensions for virtual beta orbitals
    std::vector<std::vector<int>> get_bsym_vir(int norb, std::vector<int> act_mo) const;

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

    /// Return the sign of a_n applied to this determinant
    /// this version is inefficient and should be used only for testing/debugging
    double slater_sign_safe(int n) const{
        size_t count = 0;
        for (size_t k = 0; k < n; ++k){
            if (this->get_bit(k)) count++;
        }
        return (count % 2 == 0) ? 1.0 : -1.0;
    }

    /// Return the sign of a_n applied to this determinant
    /// This function ignores if bit n is set or not
    double slater_sign(int n) const{
        size_t count = 0;
        // count all the preceeding bits only if we are looking past the first word
        if (n >= bits_per_word){
            size_t last_full_word = whichword(n);
            for (size_t k = 0; k < last_full_word; ++k){
                count += ui64_bit_count(words_[k]);
            }
        }
        return (count % 2 == 0) ?  ui64_sign(getword(n), whichbit(n)) :
                                  -ui64_sign(getword(n), whichbit(n));
    }
    /// Return the sign for a single second quantized operator
    /// This function ignores if bit n is set or not
    double slater_sign_a(int n) const {
        return slater_sign(n);
    }
    /// Return the sign for a single second quantized operator
    /// This function ignores if bit n is set or not
    double slater_sign_b(int n) const {
        return slater_sign(n + beta_offset);
    }
    /// Return the sign for a pair of second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    double slater_sign(int n, int m) const{
        // let's first order the numbers so that m <= n
        if (n < m) std::swap(m,n);
        size_t word_m = whichword(m);
        size_t word_n = whichword(n);
        // if both bits are in the same word use an optimized version
        if (word_n == word_m){
            return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
        }
        size_t count = 0;
        // count the number of bits in bitween the words of m and n
        for (size_t k = word_m + 1; k < word_n; ++k){
            count += ui64_bit_count(words_[k]);
        }
        // count the bits after m in word[m]
        // count the bits before n in word[n]
        double sign = ui64_sign_reverse(words_[word_m], whichbit(m)) *
                      ui64_sign(words_[word_n], whichbit(n));
        return (count % 2 == 0) ? sign : -sign;
    }

    /// Return the sign for a pair of alpha second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    double slater_sign_aa(int n, int m) const { return slater_sign(n,m);}

    /// Return the sign for a pair of beta second quantized operators
    /// The sign depends only on the number of bits = 1 between n and m
    double slater_sign_bb(int n, int m) const  { return slater_sign(n + beta_offset,m + beta_offset);}

    /// Return the sign for a quadruplet of alpha second quantized operators
    /// Untested, need documentation
    double slater_sign_aaaa(int i, int j, int a, int b) const  {
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

    /// Untested, need documentation
    double slater_sign_bbbb(int i, int j, int a, int b) const{
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
    /// Untested
    int count() const{
        int c = 0;
        for (auto const w : words_){
            c += ui64_bit_count(w);
        }
        return c;
    }

    int count_alfa() const {
        int na = 0;
        int w = whichword(nbits_half - 1);
        size_t b = whichbit(nbits_half - 1);
        // count the number of bits in between the words of m and n
        for (int k = 0; k < w; ++k){
            na += ui64_bit_count(words_[k]);
        }
        return na + ui64_bit_count(words_[w],b);
    }

    /// Count the number of beta bits set to true
    int count_beta() const {
        int nb = 0;
        int w = whichword(beta_offset);
        size_t b = whichbit(beta_offset);
        // count the number of bits in between the words of m and n
        for (int k = w + 1; k < nwords_; ++k){
            nb += ui64_bit_count(words_[k]);
        }
        return nb + ui64_bit_count_reverse(words_[w],b);
    };

    /// Return the number of alpha/beta pairs
    /// Untested
    int npair() const {
        int count = 0;
        for (size_t k = 0; k < nbits_half; ++k){
            if ((get_alfa_bit(k) == 1) and (get_beta_bit(k) == 1)) count += 1;
        }
        return count;
    }

    /// Flip all bits and return a reference to this object
    Determinant<N>& flip() {
        for (size_t k = 0; k < nwords_; ++k){
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
    double single_excitation_b(int i, int a){
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
    std::string str(int n = nbits_half) const {
        return str(*this, n);
    }

    /// Preliminary untested code
    struct Hash {
        std::size_t operator()(const Determinant<N>& d) const {
//          return ((bs.a_ * 13466917) + bs.b_) % 1405695061;
            std::size_t seed = d.nwords_;
            for(auto& w : d.words_) {
              seed ^= w + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // END OF FUNCTIONS NOT TESTED

private:

    // ==> Private Functions <==

    /// the index of the word where bit pos is found
    static constexpr size_t whichword(size_t pos) noexcept {
        return pos / bits_per_word;
    }
    /// the word where bit pos is found
    word_t &getword(size_t pos) { return words_[whichword(pos)]; }
    const word_t &getword(size_t pos) const { return words_[whichword(pos)]; }
    /// the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept {
        return pos % bits_per_word;
    }
    /// a mask for bit pos in its corresponding word
    static constexpr word_t maskbit(size_t pos) {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    // ==> Private Data <==

    /// The bits stored as a vector of words
#if PERFORMANCE_OPTIMIZATION
    word_t words_[nwords_];
#else
    word_t words_[nwords_] = {};
#endif
};

// Functions

/**
 * @brief return a string representation of this determinant
 */
template <size_t N>
std::string str_bits(const Determinant<N> &d, int n = Determinant<N>::nbits) {
    std::string s;
    for (int p = 0; p < n; ++p) {
        s += d.get_bit(p) ? '1' : '0';
    }
    return s;
}

template <size_t N>
std::string str(const Determinant<N> &d, int n = Determinant<N>::nbits_half) {
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

template <size_t N> void set_str(Determinant<N> &d, const std::string &str) {
    // zero all the bits and set the bits passed as a string
    d.zero();
    if (str.size() == Determinant<N>::nbits) {
        size_t k = 0;
        for (auto c : str) {
            if (c == '0') {
                d.set_bit(k, 0);
            } else {
                d.set_bit(k, 1);
            }
            k++;
        }
    }else{
        std::cout << "template <size_t N> void set_str(Determinant<N> &d, const std::string &str)";
        std::cout << "mismatch between the number of bits in d and those passed in str";
        exit(1);
    }
}

#endif // _determinant_hpp_
