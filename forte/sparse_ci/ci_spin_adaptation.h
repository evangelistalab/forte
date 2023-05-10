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

#include "sparse_ci/determinant.h"

namespace psi {
class Vector;
}

namespace forte {

class DeterminantHashVec;

/// @brief A class to perform spin adaptation of a CI wavefunction
class SpinAdapter {
  public:
    /// Class constructor
    /// @param dets A vector of determinants to be spin adapted
    SpinAdapter(int na, int nb, int twoS, int twoMs, int norb);

    /// @brief A function to prepare the determinant to CSF mapping
    /// @param dets a vector of determinants sorted according to their address
    void prepare_couplings(const std::vector<Determinant>& dets);

    /// @brief Convert a coefficient vector from the CSF basis to the determinant basis
    /// @param csf_C csf coefficients
    /// @param det_C determinant coefficients
    void csf_C_to_det_C(const std::vector<double>& csf_C, std::vector<double>& det_C);

    /// @brief Convert a coefficient vector from the determinant basis to the CSF basis
    /// @param det_C determinant coefficients
    /// @param csf_C csf coefficients
    void det_C_to_csf_C(const std::vector<double>& det_C, std::vector<double>& csf_C);

    void det_C_to_csf_C(std::shared_ptr<psi::Vector>& det_C, std::shared_ptr<psi::Vector>& csf_C);

    void csf_C_to_det_C(std::shared_ptr<psi::Vector>& csf_C, std::shared_ptr<psi::Vector>& det_C);

    /// @brief Return the number of CSFs
    size_t ncsf() const;

    /// @brief Return the number of determinants
    size_t ndet() const;

    class iterator {
      public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<size_t, double>;
        using pointer = value_type*;
        using reference = value_type&;

        iterator(pointer p) : p_(p) {}

        reference operator*() const { return *p_; }
        pointer operator->() { return p_; }

        iterator& operator++() {
            p_++;
            return *this;
        }

        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        friend bool operator==(const iterator& a, const iterator& b) { return a.p_ == b.p_; }
        friend bool operator!=(const iterator& a, const iterator& b) { return a.p_ != b.p_; }

      private:
        pointer p_;
    };

    iterator begin(size_t n) {
        if (n < csf_to_det_bounds_.size() - 1) {
            return iterator(&(csf_to_det_coeff_[csf_to_det_bounds_[n]]));
        }
        return end(n);
    }

    iterator end(size_t n) {
        if (n < csf_to_det_bounds_.size() - 1) {
            return iterator(&csf_to_det_coeff_[csf_to_det_bounds_[n + 1]]);
        }
        return iterator(csf_to_det_coeff_.data() + csf_to_det_coeff_.size());
    }

    // Custom iterable wrapper class for SA
    class CSFIterable {
      public:
        CSFIterable(SpinAdapter& sa, size_t n) : sa_(sa), n_(n) {}

        iterator begin() { return sa_.begin(n_); }
        iterator end() { return sa_.end(n_); }

      private:
        SpinAdapter& sa_;
        size_t n_;
    };

    /// @brief Return the number of determinants in a CSF
    size_t ncsf(size_t n) { return csf_to_det_bounds_[n + 1] - csf_to_det_bounds_[n]; }
    /// @brief Return an iterable object for the CSFs
    CSFIterable csf(size_t n) { return CSFIterable(*this, n); }

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

    /// @brief A function to generate all the CSFs from a configuration
    void conf_to_csfs(const Configuration& conf, int twoS, int twoMs, DeterminantHashVec& det_hash);

    /// A function to generate all possible spin couplings
    /// @param N The number of unpaired electrons
    /// @param twoS Twice the spin quantum number
    auto make_spin_couplings(int N, int twoS) -> std::vector<String>;

    /// A function to generate all possible alpha/beta occupations
    /// @param N The number of unpaired electrons
    /// @param twoS Twice the spin quantum number
    auto make_determinant_occupations(int N, int twoS) -> std::vector<String>;
};
} // namespace forte

#endif // _spin_adaptation_h_
