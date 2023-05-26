/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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

/// Spin cases for 1-, 2-, and 3-body tensors
enum class Spin1 { a, b };
enum class Spin2 { aa, ab, bb };
enum class Spin3 { aaa, aab, abb, bbb };

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
std::shared_ptr<psi::Matrix> tensor_to_matrix(ambit::Tensor t);

/**
 * @brief tensor_to_matrix
 * @param t The input tensor
 * @param dims psi::Dimensions of the matrix extracted from the tensor
 * @return A copy of the tensor data in symmetry blocked form
 */
std::shared_ptr<psi::Matrix> tensor_to_matrix(ambit::Tensor t, psi::Dimension dims);

// /**
//  * @brief view_modified_orbitals Write orbitals using molden
//  * @param Ca  The Ca matrix to be viewed with MOLDEN
//  * @param diag_F -> The Orbital energies (diagonal elements of Fock operator)
//  * @param occupation -> occupation vector
//  */
// void view_modified_orbitals(std::shared_ptr<psi::Wavefunction> wfn,
//                             const std::shared_ptr<psi::Matrix>& Ca,
//                             const std::shared_ptr<psi::Vector>& diag_F,
//                             const std::shared_ptr<psi::Vector>& occupation);

/**
 * Returns the Ms as a string, using fractions if needed
 */
std::string get_ms_string(double twice_ms);

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

// /**
//  * @brief split up a vector into different processors
//  * @param size_t size_of_tasks
//  * @param nproc (the global number of processors)
//  * @return a pair of vectors -> pair.0 -> start for each processor
//  *                           -> pair.1 -> end or each processor
//  */
// std::pair<std::vector<size_t>, std::vector<size_t>> split_up_tasks(size_t size_of_tasks,
//                                                                    size_t nproc);

template <typename T>
std::vector<std::vector<T>> split_vector(const std::vector<T>& vec, size_t max_length) {
    size_t vec_size = vec.size();
    if (max_length == 0 or vec_size == 0)
        throw std::runtime_error("Cannot split vector of size 0!");

    std::vector<std::vector<T>> out_vec;

    size_t n_even = vec_size / max_length;
    for (size_t i = 0, begin = 0, end = max_length; i < n_even; ++i) {
        out_vec.push_back(std::vector<T>(vec.begin() + begin, vec.begin() + end));
        begin = end;
        end += max_length;
    }

    if (vec_size % max_length) {
        out_vec.push_back(std::vector<T>(vec.begin() + n_even * max_length, vec.end()));
    }

    return out_vec;
}

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

/**
 * @brief Apply in-place matrix transposition based on the algorithm of Catanzaro, Keller, Garland.
 * @param data the matrix stored in row-major format
 * @param m the number of rows of the matrix
 * @param n the number of columns of the matrix
 *
 * See Algorithm 1 of DOI: 10.1145/2555243.2555253.
 * Also see https://github.com/bryancatanzaro/inplace
 */
void matrix_transpose_in_place(std::vector<double>& data, const size_t m, const size_t n);

void push_to_psi4_env_globals(double value, const std::string& label);

namespace math {
/// Return the number of combinations of n identical objects
size_t combinations(size_t n, size_t k);

/// Return the Cartesian product of the input vector<vector<T>>
/// https://stackoverflow.com/a/17050528/4101036
template <typename T>
std::vector<std::vector<T>> cartesian_product(const std::vector<std::vector<T>>& input) {
    std::vector<std::vector<T>> product{{}};

    for (const auto& vec : input) {
        std::vector<std::vector<T>> tmp;
        for (const auto& x : product) {
            for (const auto& y : vec) {
                tmp.push_back(x);
                tmp.back().push_back(y);
            }
        }
        product = std::move(tmp);
    }

    return product;
}
} // namespace math

} // namespace forte

#endif // _helpers_h_
