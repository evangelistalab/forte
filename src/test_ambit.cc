/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "test_ambit.h"

#include <cstdlib>

namespace psi {
namespace forte {

AMBIT_TEST::AMBIT_TEST() {}

AMBIT_TEST::~AMBIT_TEST() {}

Tensor AMBIT_TEST::build_and_fill(const std::string& name, const ambit::Dimension& dims,
                                  double matrix[MAXTWO]) {
    Tensor T = Tensor::build(CoreTensor, name, dims);
    initialize_random(T, matrix);
    std::pair<double, double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero)
        throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor AMBIT_TEST::build_and_fill(const std::string& name, const ambit::Dimension& dims,
                                  double matrix[MAXTWO][MAXTWO]) {
    Tensor T = Tensor::build(CoreTensor, name, dims);
    initialize_random(T, matrix);
    std::pair<double, double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero)
        throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor AMBIT_TEST::build_and_fill(const std::string& name, const ambit::Dimension& dims,
                                  double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    Tensor T = Tensor::build(CoreTensor, name, dims);
    initialize_random(T, matrix);
    std::pair<double, double> a_diff = difference(T, matrix);
    if (std::fabs(a_diff.second) > zero)
        throw std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

void AMBIT_TEST::initialize_random(Tensor& tensor, double matrix[MAXTWO]) {
    std::srand(0);
    size_t n0 = tensor.dims()[0];
    std::vector<double>& vec = tensor.data();
    for (size_t i = 0; i < n0; ++i) {
        double randnum = double(std::rand()) / double(RAND_MAX);
        matrix[i] = randnum;
        vec[i] = randnum;
    }
}

void AMBIT_TEST::initialize_random(Tensor& tensor, double matrix[MAXTWO][MAXTWO]) {
    std::srand(0);
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    std::vector<double>& vec = tensor.data();
    for (size_t i = 0, ij = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j, ++ij) {
            double randnum = double(std::rand()) / double(RAND_MAX);
            matrix[i][j] = randnum;
            vec[ij] = randnum;
        }
    }
}

std::pair<double, double> AMBIT_TEST::difference(Tensor& tensor, double matrix[MAXTWO]) {
    size_t n0 = tensor.dims()[0];

    const std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < n0; ++i) {
        double diff = std::fabs(matrix[i] - result[i]);
        sum_diff += diff;
        max_diff = std::max(diff, max_diff);
    }
    return std::make_pair(sum_diff, max_diff);
}

std::pair<double, double> AMBIT_TEST::difference(Tensor& tensor, double matrix[MAXTWO][MAXTWO]) {
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];

    const std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0, ij = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j, ++ij) {
            double diff = std::fabs(matrix[i][j] - result[ij]);
            sum_diff += diff;
            max_diff = std::max(diff, max_diff);
        }
    }
    return std::make_pair(sum_diff, max_diff);
}

void AMBIT_TEST::initialize_random(Tensor& tensor,
                                   double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    std::srand(0);
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    std::vector<double>& vec = tensor.data();
    for (size_t i = 0, ijkl = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j) {
            for (size_t k = 0; k < n2; ++k) {
                for (size_t l = 0; l < n3; ++l, ++ijkl) {
                    double randnum = double(std::rand()) / double(RAND_MAX);
                    matrix[i][j][k][l] = randnum;
                    vec[ijkl] = randnum;
                }
            }
        }
    }
}

std::pair<double, double>
AMBIT_TEST::difference(Tensor& tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    const std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;

    for (size_t i = 0, ijkl = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j) {
            for (size_t k = 0; k < n2; ++k) {
                for (size_t l = 0; l < n3; ++l, ++ijkl) {
                    double diff = std::fabs(matrix[i][j][k][l] - result[ijkl]);
                    sum_diff += diff;
                    max_diff = std::max(diff, max_diff);
                }
            }
        }
    }
    return std::make_pair(sum_diff, max_diff);
}

double AMBIT_TEST::test_Cij_equal_Aik_Bkj(size_t ni, size_t nj, size_t nk) {

    Tensor A = build_and_fill("A", {ni, nk}, a2);
    Tensor B = build_and_fill("B", {nk, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k) {
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }

    return difference(C, c2).second;
}

double AMBIT_TEST::compute_energy() {
    outfile->Printf("\n\n==> Tests ambit <==\n\n");

    size_t ni = 54, nj = 63, nk = 42;
    double error = test_Cij_equal_Aik_Bkj(ni, nj, nk);

    outfile->Printf(
        "  C(\"ij\") = A(\"ik\") * B(\"kj\"): ni=%d, nj=%d, nk=%d  error=%.2e %s\n", ni, nj,
        nk, fabs(error), fabs(error) > zero ? "Failed" : "Passed");

    return 0.0;
}
}
}
