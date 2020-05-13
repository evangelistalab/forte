/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _helpers_h_
#define _helpers_h_

#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libqt/qt.h"

namespace py = pybind11;

class Options;

namespace forte {

/**
 * @brief Convert an ambit tensor to a numpy ndarray.
 *        The returned tensor is stored according to the C storage convention.
 * @param t The input tensor
 * @return A numpy array
 */
py::array_t<double> ambit_to_np(ambit::Tensor t);

/**
 * @brief Convert a std::vector<double> to a numpy ndarray.
 *        The returned tensor is stored according to the C storage convention.
 * @param v The input vector
 * @param dims The dimensions of the tensor
 * @return A numpy array
 */
py::array_t<double> vector_to_np(const std::vector<double>& v, const std::vector<size_t>& dims);
py::array_t<double> vector_to_np(const std::vector<double>& v, const std::vector<int>& dims);

/**
 * @brief tensor_to_matrix
 * @param t The input tensor
 * @return A copy of the tensor data ignoring symmetry blocks
 */
psi::SharedMatrix tensor_to_matrix(ambit::Tensor t);

/**
 * @brief tensor_to_matrix
 * @param t The input tensor
 * @param dims psi::Dimensions of the matrix extracted from the tensor
 * @return A copy of the tensor data in symmetry blocked form
 */
psi::SharedMatrix tensor_to_matrix(ambit::Tensor t, psi::Dimension dims);

/**
 * @brief view_modified_orbitals Write orbitals using molden
 * @param Ca  The Ca matrix to be viewed with MOLDEN
 * @param diag_F -> The Orbital energies (diagonal elements of Fock operator)
 * @param occupation -> occupation vector
 */
void view_modified_orbitals(std::shared_ptr<psi::Wavefunction> wfn,
                            const std::shared_ptr<psi::Matrix>& Ca,
                            const std::shared_ptr<psi::Vector>& diag_F,
                            const std::shared_ptr<psi::Vector>& occupation);

/**
 * Returns the Ms as a string, using fractions if needed
 */
std::string get_ms_string(double twice_ms);

std::string to_string(const std::vector<std::string>& vec_str, const std::string& sep = ",");

/**
 * @brief Compute the memory (in GB) required to store arrays
 * @typename T The data typename
 * @param num_el The number of elements to store
 * @return The size in GB
 */
template <typename T> double to_gb(T num_el) {
    return static_cast<double>(num_el) * static_cast<double>(sizeof(T)) / 1073741824.0;
}

/**
 * @brief Compute the memory requirement
 * @param nele The number of elements for storage
 * @param type_size The size of the data type
 * @return A pair of size in appropriate unit (B, KB, MB, GB, TB, PB)
 */
std::pair<double, std::string> to_xb(size_t nele, size_t type_size);

/**
 * @brief split up a vector into different processors
 * @param size_t size_of_tasks
 * @param nproc (the global number of processors)
 * @return a pair of vectors -> pair.0 -> start for each processor
 *                           -> pair.1 -> end or each processor
 */
std::pair<std::vector<size_t>, std::vector<size_t>> split_up_tasks(size_t size_of_tasks,
                                                                   size_t nproc);

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec, Compare& compare) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
void apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p) {
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

} // namespace forte

#endif // _helpers_h_
