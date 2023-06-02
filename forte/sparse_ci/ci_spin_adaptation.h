/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _spin_adaptation_h_
#define _spin_adaptation_h_

#include <vector>

namespace psi {
class Vector;
}

namespace forte {

class DeterminantHashVec;

/// @brief A class to perform spin adaptation of a CI wavefunction
///
/// This class is used to convert a CI wavefunction from the determinant basis to the CSF basis
/// and vice versa.  The CSF basis is defined by the spin quantum number (S) and the spin
/// projection quantum number (Ms) of the target state.  The determinant basis is defined by the
/// number of electrons (N) and the number of orbitals (norbs).  The conversion is performed by
/// using the mapping between the CSF basis and the determinant basis.
///
/// To use this class, first create an object of this class by specifying the spin quantum number
/// (S) and the spin projection quantum number (Ms) of the target state.  Then, call the
/// prepare_couplings() function to prepare the mapping between the CSF basis and the determinant
/// basis.  Finally, call the csf_C_to_det_C() function to convert the CI coefficients from the
/// CSF basis to the determinant basis, or call the det_C_to_csf_C() function to convert the CI
/// coefficients from the determinant basis to the CSF basis.
/// For example:
/// @code
/// int twoS = 0; // singlet
/// int twoMs = 0; // Ms = 0
/// int norbs = 10; // 10 orbitals
////
/// SpinAdapter sa(twoS, twoMs, norbs);
/// std::vector<Determinant> dets;
///
/// // fill dets with determinants
///
/// sa.prepare_couplings(dets);
/// size_t ndet = sa.ndet();
/// size_t ncsf = sa.ncsf();
/// std::shared_ptr<psi::Vector> det_C = std::make_shared<psi::Vector>(dets.size());
/// std::shared_ptr<psi::Vector> csf_C = std::make_shared<psi::Vector>(ncsf);
///
/// // fill det_C with coefficients in the determinant basis
///
/// sa.det_C_to_csf_C(det_C, csf_C); // convert det_C to csf_C
///
/// // do something with csf_C
///
/// sa.csf_C_to_det_C(csf_C, det_C); // convert csf_C back to det_C

class SpinAdapter {
  public:
    /// Class constructor
    /// @param twoS twice the spin quantum number (S) of the target state
    /// @param twoMs twice the spin projection quantum number (Ms) of the target state
    /// @param norb number of orbitals
    SpinAdapter(int twoS, int twoMs, int norb);

    /// @brief A function to prepare the determinant to CSF mapping
    /// @param dets a vector of determinants sorted according to their address
    void prepare_couplings(const std::vector<Determinant>& dets);

    /// @brief Convert a coefficient vector from the CSF basis to the determinant basis
    /// @param csf_C csf coefficients
    /// @param det_C determinant coefficients
    void csf_C_to_det_C(std::shared_ptr<psi::Vector>& csf_C, std::shared_ptr<psi::Vector>& det_C);

    /// @brief Convert a coefficient vector from the determinant basis to the CSF basis
    /// @param det_C determinant coefficients
    /// @param csf_C csf coefficients
    void det_C_to_csf_C(std::shared_ptr<psi::Vector>& det_C, std::shared_ptr<psi::Vector>& csf_C);

    /// @brief Return the number of CSFs
    size_t ncsf() const;

    /// @brief Return the number of determinants
    size_t ndet() const;

    /// @brief An const interator for the expansion coefficients of a CSF in the determinant
    /// basis
    class const_iterator {
      public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<size_t, double>;
        using pointer = const value_type*;
        using reference = const value_type&;

        // iterator initialization
        const_iterator(pointer p) : p_(p) {}
        // iterator dereferencing
        reference operator*() const { return *p_; }
        // iterator arrow operator
        pointer operator->() { return p_; }
        // prefix increment
        const_iterator& operator++() {
            p_++;
            return *this;
        }
        // postfix increment
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }
        // comparison operators
        friend bool operator==(const const_iterator& a, const const_iterator& b) {
            return a.p_ == b.p_;
        }
        friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return a.p_ != b.p_;
        }

      private:
        pointer p_; // pointer to the current element
    };

    /// @brief Return an iterator to the beginning of the n-th CSF in the determinant basis
    const_iterator begin(size_t n) const {
        if (n < csf_to_det_bounds_.size() - 1) {
            return const_iterator(&(csf_to_det_coeff_[csf_to_det_bounds_[n]]));
        }
        return end(n);
    }

    /// @brief Return an iterator to the end of the n-th CSF in the determinant basis
    const_iterator end(size_t n) const {
        if (n < csf_to_det_bounds_.size() - 1) {
            return const_iterator(&csf_to_det_coeff_[csf_to_det_bounds_[n + 1]]);
        }
        return const_iterator(csf_to_det_coeff_.data() + csf_to_det_coeff_.size());
    }

    /// Custom iterable wrapper class for SpinAdapter
    class CSFIterable {
      public:
        /// @brief Construct a CSFIterable object
        /// @param sa SpinAdapter object
        /// @param n the CSF index
        CSFIterable(const SpinAdapter& sa, size_t n) : sa_(sa), n_(n) {}

        const_iterator begin() { return sa_.begin(n_); }
        const_iterator end() { return sa_.end(n_); }

      private:
        const SpinAdapter& sa_;
        const size_t n_;
    };

    /// @brief Return the number of determinants in a CSF
    size_t ncsf(size_t n) const { return csf_to_det_bounds_[n + 1] - csf_to_det_bounds_[n]; }
    /// @brief Return an iterable object for the CSFs
    CSFIterable csf(size_t n) const { return CSFIterable(*this, n); }

  private:
    /// @brief Twice the spin quantum number (2S)
    int twoS_ = 0;
    /// @brief Twice the spin projection quantum number (2Ms)
    int twoMs_ = 0;
    /// @brief The number of orbitals
    int norb_ = 0;
    /// @brief The number of CSFs
    size_t ncsf_ = 0;
    /// @brief The number of determinants
    size_t ndet_ = 0;
    /// @brief The number of spin couplings
    size_t ncoupling_ = 0;
    /// @brief A vector with the starting index of each CSF in the determinant basis
    std::vector<size_t> csf_to_det_bounds_;
    /// @brief A vector used to store information on how to map the CSFs to determinants
    std::vector<std::pair<size_t, double>> csf_to_det_coeff_;
    /// @brief A vector used to store the configurations
    std::vector<Configuration> confs_;

    /// @bried A vector with the number of CSFs with a given number of unpaired electrons (N)
    std::vector<size_t> N_ncsf_;
    /// @brief A vector used to store the determinant occupations for N unpaired electrons
    std::vector<std::vector<String>> N_to_det_occupations_;
    /// @brief A vector used to store the overlap between CSFs and determinants
    std::vector<std::vector<std::tuple<size_t, size_t, double>>> N_to_overlaps_;
    /// @brief Stores the number of non-zero overlaps there are for each N and spin coupling
    std::vector<std::vector<size_t>> N_to_noverlaps_;

    /// @brief Compute the unique spin couplings
    /// @returns the number of couplings and CSFs
    auto compute_unique_couplings();

    /// @brief A function to generate all the CSFs from a configuration
    void conf_to_csfs(const Configuration& conf, DeterminantHashVec& det_hash);

    /// @brief A function to generate all possible spin couplings stored as strings. The spin
    /// couplings are stored in String objects with the following format:
    /// up coupling = 0, down coupling = 1
    /// @param N The number of unpaired electrons
    /// @param twoS Twice the spin quantum number
    auto make_spin_couplings(int N, int twoS) -> std::vector<String>;

    /// A function to generate all possible alpha/beta occupations stored as strings. The
    /// occupations are stored in String objects with the following format:
    /// up spin (alpha) = 0, down spin (beta) = 1
    /// @param N The number of unpaired electrons
    /// @param twoS Twice the spin quantum number
    auto make_determinant_occupations(int N, int twoS) -> std::vector<String>;
};
} // namespace forte

#endif // _spin_adaptation_h_
