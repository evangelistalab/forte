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
#define DEBUG_SPIN_ADAPTATION 1

namespace forte {

/// @brief A class to perform spin adaptation on a CI wavefunction
class SpinAdapter {
  public:
    /// Class constructor
    /// @param dets A vector of determinants to be spin adapted
    SpinAdapter(int na, int nb, int twoS, int twoMs, int norb);

    void prepare_couplings(const std::vector<Determinant>& dets);

    void csf_C_to_det_C(const std::vector<double>& csf_C, std::vector<double>& det_C);

    void det_C_to_csf_C(const std::vector<double>& det_C, std::vector<double>& csf_C);

    /// @brief A function to generate all the CSFs from a configuration
    auto conf_to_csfs(const Configuration& conf, int twoS, int twoMs)
        -> std::vector<std::vector<double>>;

  private:
    /// @brief The number of alpha electrons
    int na_;
    /// @brief The number of beta electrons
    int nb_;
    /// @brief Twice the spin quantum number (2S)
    int twoS_;
    /// @brief Twice the spin projection quantum number (2Ms)
    int twoMs_;
    /// @brief The number of orbitals
    int norb_;
    /// @brief A vector with the number of CSFs that contribute to a determinant
    std::vector<size_t> det_to_csf_size_;
    /// @brief A vector with the number of determinants that contribute to a CSF
    std::vector<size_t> csf_to_det_size_;
    /// @brief A vector used to store information on how to map the determinants to CSFs
    std::vector<std::tuple<size_t, double>> det_to_csf_coeff_;
    /// @brief A vector used to store information on how to map the CSFs to determinants
    std::vector<std::tuple<size_t, double>> csf_to_det_coeff_;
    /// @brief A vector used to store the configurations
    std::vector<Configuration> confs_;

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
