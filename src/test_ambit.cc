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
#include <limits>

namespace psi {
namespace forte {
namespace AMBIT_TEST {

#include <ambit/tensor.h>
#include <cstring>
#include <cstdlib>
//#include <cstdio>
#include <cmath>
//#include <utility>
#include <stdexcept>

#define MAXTWO 10
#define MAXFOUR 10

//#define ANSI_COLOR_RED "\x1b[31m"
//#define ANSI_COLOR_GREEN "\x1b[32m"
//#define ANSI_COLOR_YELLOW "\x1b[33m"
//#define ANSI_COLOR_BLUE "\x1b[34m"
//#define ANSI_COLOR_MAGENTA "\x1b[35m"
//#define ANSI_COLOR_CYAN "\x1b[36m"
//#define ANSI_COLOR_RESET "\x1b[0m"

double a1[MAXTWO];
double a2[MAXTWO][MAXTWO];
double b2[MAXTWO][MAXTWO];
double c2[MAXTWO][MAXTWO];
double d2[MAXTWO][MAXTWO];
double e2[MAXTWO][MAXTWO];
double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double e4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

using namespace ambit;

enum TestResult { kPass, kFail, kException };

/// Initialize a tensor and a 2-dim matrix with random numbers
void initialize_random(Tensor& tensor, double matrix[MAXTWO]);
::std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO]);

void initialize_random(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);
::std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);

void initialize_random(Tensor& tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
::std::pair<double, double> difference(Tensor& tensor,
                                       double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

double zero = 1.0e-09;

TensorType tensor_type = CoreTensor;

Tensor build_and_fill(const ::std::string& name, const ambit::Dimension& dims,
                      double matrix[MAXTWO]) {
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    ::std::pair<double, double> a_diff = difference(T, matrix);
    if (::std::fabs(a_diff.second) > zero)
        throw ::std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor build_and_fill(const ::std::string& name, const ambit::Dimension& dims,
                      double matrix[MAXTWO][MAXTWO]) {
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    ::std::pair<double, double> a_diff = difference(T, matrix);
    if (::std::fabs(a_diff.second) > zero)
        throw ::std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

Tensor build_and_fill(const ::std::string& name, const ambit::Dimension& dims,
                      double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    Tensor T = Tensor::build(tensor_type, name, dims);
    initialize_random(T, matrix);
    ::std::pair<double, double> a_diff = difference(T, matrix);
    if (::std::fabs(a_diff.second) > zero)
        throw ::std::runtime_error("Tensor and standard matrix don't match.");
    return T;
}

void initialize_random(Tensor& tensor, double matrix[MAXTWO]) {
    ::std::srand(0);
    size_t n0 = tensor.dims()[0];
    ::std::vector<double>& vec = tensor.data();
    for (size_t i = 0; i < n0; ++i) {
        double randnum = double(::std::rand()) / double(RAND_MAX);
        matrix[i] = randnum;
        vec[i] = randnum;
    }
}

void initialize_random(Tensor& tensor, double matrix[MAXTWO][MAXTWO]) {
    ::std::srand(0);
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    ::std::vector<double>& vec = tensor.data();
    for (size_t i = 0, ij = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j, ++ij) {
            double randnum = double(::std::rand()) / double(RAND_MAX);
            matrix[i][j] = randnum;
            vec[ij] = randnum;
        }
    }
}

::std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO]) {
    size_t n0 = tensor.dims()[0];

    const ::std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0; i < n0; ++i) {
        double diff = ::std::fabs(matrix[i] - result[i]);
        sum_diff += diff;
        max_diff = ::std::max(diff, max_diff);
    }
    return ::std::make_pair(sum_diff, max_diff);
}

::std::pair<double, double> difference(Tensor& tensor, double matrix[MAXTWO][MAXTWO]) {
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];

    const ::std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;
    for (size_t i = 0, ij = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j, ++ij) {
            double diff = ::std::fabs(matrix[i][j] - result[ij]);
            sum_diff += diff;
            max_diff = ::std::max(diff, max_diff);
        }
    }
    return ::std::make_pair(sum_diff, max_diff);
}

void initialize_random(Tensor& tensor, double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    ::std::srand(0);
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    ::std::vector<double>& vec = tensor.data();
    for (size_t i = 0, ijkl = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j) {
            for (size_t k = 0; k < n2; ++k) {
                for (size_t l = 0; l < n3; ++l, ++ijkl) {
                    double randnum = double(::std::rand()) / double(RAND_MAX);
                    matrix[i][j][k][l] = randnum;
                    vec[ijkl] = randnum;
                }
            }
        }
    }
}

::std::pair<double, double> difference(Tensor& tensor,
                                       double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]) {
    size_t n0 = tensor.dims()[0];
    size_t n1 = tensor.dims()[1];
    size_t n2 = tensor.dims()[2];
    size_t n3 = tensor.dims()[3];

    const ::std::vector<double>& result = tensor.data();

    double sum_diff = 0.0;
    double max_diff = 0.0;

    for (size_t i = 0, ijkl = 0; i < n0; ++i) {
        for (size_t j = 0; j < n1; ++j) {
            for (size_t k = 0; k < n2; ++k) {
                for (size_t l = 0; l < n3; ++l, ++ijkl) {
                    double diff = ::std::fabs(matrix[i][j][k][l] - result[ijkl]);
                    sum_diff += diff;
                    max_diff = ::std::max(diff, max_diff);
                }
            }
        }
    }
    return ::std::make_pair(sum_diff, max_diff);
}

double test_C_equal_A_B(::std::string c_ind, ::std::string a_ind, ::std::string b_ind,
                        ::std::vector<int> c_dim, ::std::vector<int> a_dim,
                        ::std::vector<int> b_dim) {
    ::std::vector<size_t> dims;
    dims.push_back(9);
    dims.push_back(6);
    dims.push_back(7);

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Tensor A = build_and_fill("A", {dims[a_dim[0]], dims[a_dim[1]]}, a2);
    Tensor B = build_and_fill("B", {dims[b_dim[0]], dims[b_dim[1]]}, b2);
    Tensor C = build_and_fill("C", {dims[c_dim[0]], dims[c_dim[1]]}, c2);

    C(c_ind) += A(a_ind) * B(b_ind);

    ::std::vector<size_t> n(3);
    for (n[0] = 0; n[0] < ni; ++n[0]) {
        for (n[1] = 0; n[1] < nj; ++n[1]) {
            for (n[2] = 0; n[2] < nk; ++n[2]) {
                size_t aind1 = n[a_dim[0]];
                size_t aind2 = n[a_dim[1]];
                size_t bind1 = n[b_dim[0]];
                size_t bind2 = n[b_dim[1]];
                size_t cind1 = n[c_dim[0]];
                size_t cind2 = n[c_dim[1]];
                c2[cind1][cind2] += a2[aind1][aind2] * b2[bind1][bind2];
            }
        }
    }

    return difference(C, c2).second;
}

double test_wrapper() {
    double max_error = 0.0, current_error = 0.0;

    current_error = test_C_equal_A_B("ij", "ik", "jk", {0, 1}, {0, 2}, {1, 2});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ij", "ik", "kj", {0, 1}, {0, 2}, {2, 1});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ij", "ki", "jk", {0, 1}, {2, 0}, {1, 2});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ij", "ki", "kj", {0, 1}, {2, 0}, {2, 1});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ji", "ik", "jk", {1, 0}, {0, 2}, {1, 2});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ji", "ik", "kj", {1, 0}, {0, 2}, {2, 1});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ji", "ki", "jk", {1, 0}, {2, 0}, {1, 2});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);
    current_error = test_C_equal_A_B("ji", "ki", "kj", {1, 0}, {2, 0}, {2, 1});
    if (::std::fabs(current_error) > max_error)
        max_error = ::std::fabs(current_error);

    return max_error;
}

double test_Cij_plus_equal_Aik_Bkj() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Tensor A = build_and_fill("A", {ni, nk}, a2);
    Tensor B = build_and_fill("B", {nk, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") += A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }

    return difference(C, c2).second;
}

double test_Cij_minus_equal_Aik_Bkj() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Tensor A = build_and_fill("A", {ni, nk}, a2);
    Tensor B = build_and_fill("B", {nk, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") -= A("ik") * B("kj");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                c2[i][j] -= a2[i][k] * b2[k][j];
            }
        }
    }

    return difference(C, c2).second;
}

double test_Cij_equal_Aik_Bkj() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Tensor A = build_and_fill("A", {ni, nk}, a2);
    Tensor B = build_and_fill("B", {nk, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C.zero();
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

double test_Cij_equal_Aik_Bjk() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;

    Tensor A = build_and_fill("A", {ni, nk}, a2);
    Tensor B = build_and_fill("B", {nj, nk}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = A("ik") * B("jk");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k) {
                c2[i][j] += a2[i][k] * b2[j][k];
            }
        }
    }

    return difference(C, c2).second;
}

double test_Cijkl_equal_Aijab_Bklab() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Tensor A = build_and_fill("A", {ni, nj, na, nb}, a4);
    Tensor B = build_and_fill("B", {nk, nl, na, nb}, b4);
    Tensor C = build_and_fill("C", {ni, nj, nk, nl}, c4);

    C("ijkl") += A("ijab") * B("klab");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t a = 0; a < na; ++a) {
                        for (size_t b = 0; b < nb; ++b) {
                            c4[i][j][k][l] += a4[i][j][a][b] * b4[k][l][a][b];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cikjl_equal_Aijab_Bklab() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Tensor A = build_and_fill("A", {ni, nj, na, nb}, a4);
    Tensor B = build_and_fill("B", {nk, nl, na, nb}, b4);
    Tensor C = build_and_fill("C", {ni, nk, nj, nl}, c4);

    C("ikjl") += A("ijab") * B("klab");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t a = 0; a < na; ++a) {
                        for (size_t b = 0; b < nb; ++b) {
                            c4[i][k][j][l] += a4[i][j][a][b] * b4[k][l][a][b];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cij_equal_Aiabc_Bjabc() {
    size_t ni = 9;
    size_t nj = 6;
    size_t na = 6;
    size_t nb = 7;
    size_t nc = 8;

    Tensor A = build_and_fill("A", {ni, na, nb, nc}, a4);
    Tensor B = build_and_fill("B", {nj, na, nb, nc}, b4);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") += A("iabc") * B("jabc");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t a = 0; a < na; ++a) {
                for (size_t b = 0; b < nb; ++b) {
                    for (size_t c = 0; c < nc; ++c) {
                        c2[i][j] += a4[i][a][b][c] * b4[j][a][b][c];
                    }
                }
            }
        }
    }

    return difference(C, c2).second;
}

double test_Cij_minus_equal_Aij_Bij() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") -= A("ij") * B("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] -= a2[i][j] * b2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_Dij_plus_equal_Aij_Bij_Cij() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);
    Tensor D = build_and_fill("C", {ni, nj}, d2);

    D("ij") += A("ij") * B("ij") * C("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] += a2[i][j] * b2[i][j] * c2[i][j];
        }
    }

    return difference(D, d2).second;
}

double test_E_abcd_equal_Aijab_Bklcd_C_jl_D_ik() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;
    size_t nc = 6;
    size_t nd = 7;

    Tensor A = build_and_fill("A", {ni, nj, na, nb}, a4);
    Tensor B = build_and_fill("B", {nk, nl, nc, nd}, b4);
    Tensor C = build_and_fill("C", {nj, nl}, c2);
    Tensor D = build_and_fill("D", {ni, nk}, d2);
    Tensor E = build_and_fill("E", {na, nb, nc, nd}, e4);

    E("abcd") += A("ijab") * B("klcd") * C("jl") * D("ik");

    for (size_t a = 0; a < na; ++a) {
        for (size_t b = 0; b < nb; ++b) {
            for (size_t c = 0; c < nc; ++c) {
                for (size_t d = 0; d < nd; ++d) {
                    for (size_t i = 0; i < ni; ++i) {
                        for (size_t j = 0; j < nj; ++j) {
                            for (size_t k = 0; k < nk; ++k) {
                                for (size_t l = 0; l < nl; ++l) {
                                    e4[a][b][c][d] +=
                                        a4[i][j][a][b] * b4[k][l][c][d] * c2[j][l] * d2[i][k];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return difference(E, e4).second;
}

double test_C_equal_2_A() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = 2.0 * a2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_C_plus_equal_2_A() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") += 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] += 2.0 * a2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_C_minus_equal_2_A() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") -= 2.0 * A("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] -= 2.0 * a2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_C_times_equal_2() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") *= 2.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] *= 2.0;
        }
    }

    return difference(C, c2).second;
}

double test_C_divide_equal_2() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") /= 2.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] /= 2.0;
        }
    }

    return difference(C, c2).second;
}

double test_Cij_equal_Aji() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {nj, ni}, a2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = A("ji");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = a2[j][i];
        }
    }

    return difference(C, c2).second;
}

double test_Cijkl_equal_Akijl() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 5;
    size_t nl = 4;

    Tensor A = build_and_fill("A", {nk, ni, nj, nl}, a4);
    Tensor C = build_and_fill("C", {ni, nj, nk, nl}, c4);

    C("ijkl") = A("kijl");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    c4[i][j][k][l] = a4[k][i][j][l];
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cijkl_equal_Akilj() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 5;
    size_t nl = 4;

    Tensor A = build_and_fill("A", {nk, ni, nl, nj}, a4);
    Tensor C = build_and_fill("C", {ni, nj, nk, nl}, c4);

    C("ijkl") = A("kilj");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    c4[i][j][k][l] = a4[k][i][l][j];
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cij_equal_Cij() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor C = build_and_fill("C", {ni, nj}, c2);

    // This is expected to throw. Caught above.
    C("ij") = C("ij");

    return 0.0;
}

double test_syev() {
    size_t ni = 9;

    Tensor C = build_and_fill("C", {ni, ni}, c2);

    auto result = C.syev(DescendingEigenvalue);

    //    Tensor vectors = result["eigenvectors"];

    //    result["eigenvectors"].print(stdout, 1);
    //    result["eigenvalues"].print(stdout, 1);

    return 0.0;
}

double test_geev() {
    size_t ni = 9;

    Tensor C = build_and_fill("C", {ni, ni}, c2);

    auto result = C.geev(AscendingEigenvalue);

    //    Tensor vectors = result["eigenvectors"];

    //    result["v"].print(stdout, 1);
    //    result["u"].print(stdout, 1);
    //    result["lambda"].print(stdout, 1);
    //    result["lambda i"].print(stdout, 1);

    return 0.0;
}

double test_Cilkj_equal_Aibaj_Bblak() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Tensor A = build_and_fill("A", {ni, nb, na, nj}, a4);
    Tensor B = build_and_fill("B", {nb, nl, na, nk}, b4);
    Tensor C = build_and_fill("C", {ni, nl, nk, nj}, c4);

    C("ilkj") += A("ibaj") * B("blak");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t a = 0; a < na; ++a) {
                        for (size_t b = 0; b < nb; ++b) {
                            c4[i][l][k][j] += a4[i][b][a][j] * b4[b][l][a][k];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cljik_equal_Abija_Blbak() {
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t na = 6;
    size_t nb = 7;

    Tensor A = build_and_fill("A", {nb, ni, nj, na}, a4);
    Tensor B = build_and_fill("B", {nl, nb, na, nk}, b4);
    Tensor C = build_and_fill("C", {nl, nj, ni, nk}, c4);

    C("ljik") += A("bija") * B("lbak");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t a = 0; a < na; ++a) {
                        for (size_t b = 0; b < nb; ++b) {
                            c4[l][j][i][k] += a4[b][i][j][a] * b4[l][b][a][k];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_Cij_equal_Aij_plus_Bij() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = A("ij") + B("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = a2[i][j] + b2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_Dij_equal_Aij_plus_Bij_plus_Cij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);
    Tensor D = build_and_fill("D", dims, d2);

    D("ij") = A("ij") + B("ij") + C("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] = a2[i][j] + b2[i][j] + c2[i][j];
        }
    }

    return difference(D, d2).second;
}

double test_Cij_equal_Aij_minus_Bij() {
    size_t ni = 9;
    size_t nj = 6;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nj}, b2);
    Tensor C = build_and_fill("C", {ni, nj}, c2);

    C("ij") = A("ij") - 5.0 * B("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = a2[i][j] - 5.0 * b2[i][j];
        }
    }

    return difference(C, c2).second;
}

double test_Dij_equal_Aij_minus_Bij_plus_Cij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);
    Tensor D = build_and_fill("D", dims, d2);

    D("ij") = A("ij") - B("ij") + 2.0 * C("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] = a2[i][j] - b2[i][j] + 2.0 * c2[i][j];
        }
    }

    return difference(D, d2).second;
}

double test_Dij_equal_Aij_times_Bij_plus_Cij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);
    Tensor D = build_and_fill("D", dims, d2);

    D("ij") = A("ij") * (2.0 * B("ij") - C("ij"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] = a2[i][j] * (2.0 * b2[i][j] - c2[i][j]);
        }
    }

    return difference(D, d2).second;
}

double test_Dij_equal_Bij_plus_Cij_times_Aij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);
    Tensor D = build_and_fill("D", dims, d2);

    D("ij") = (2.0 * B("ij") - C("ij")) * A("ij");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] = (2.0 * b2[i][j] - c2[i][j]) * a2[i][j];
        }
    }

    return difference(D, d2).second;
}

double test_F_equal_D_times_2g_minus_g() {
    size_t ni = 9, nj = 9, nk = 9, nl = 9;

    Tensor F = build_and_fill("F", {ni, nj}, a2);
    Tensor D = build_and_fill("D", {nk, nl}, b2);
    Tensor g = build_and_fill("g", {ni, nj, nk, nl}, c4);

    F("i,j") = D("k,l") * (2.0 * g("i,j,k,l") - g("i,k,j,l"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            a2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    a2[i][j] += b2[k][l] * (2.0 * c4[i][j][k][l] - c4[i][k][j][l]);
                }
            }
        }
    }

    return difference(F, a2).second;
}

double test_Dij_equal_2_times_Aij_plus_Bij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);

    C("ij") = 2.0 * (A("ij") - B("ij"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = 2.0 * (a2[i][j] - b2[i][j]);
        }
    }

    return difference(C, c2).second;
}

double test_Dij_equal_negate_Aij_plus_Bij() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);

    C("ij") = -(A("ij") + B("ij"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c2[i][j] = -(a2[i][j] + b2[i][j]);
        }
    }

    return difference(C, c2).second;
}

double test_power() {
    size_t ni = 9;

    Tensor C = build_and_fill("C", {ni, ni}, c2);

    Tensor A = C.power(-0.5);

    //    A.print(stdout, true);

    return 0;
}

double test_dot_product() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);

    double C = A("ij") * B("ij");
    double c = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c += a2[i][j] * b2[i][j];
        }
    }

    return ::std::fabs(C - c);
}

double test_dot_product2() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);

    double C = A("ij") * B("ik");
    double c = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c += a2[i][j] * b2[i][j];
        }
    }

    return ::std::fabs(C - c);
}

double test_dot_product3() {
    size_t ni = 9, nj = 6, nk = 5;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nk}, b2);

    // Test if the user attempts to use the correct indices with wrong
    // ambit::Dimensions
    double C = A("ij") * B("ij");
    double c = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            c += a2[i][j] * b2[i][j];
        }
    }

    return ::std::fabs(C - c);
}

double test_dot_product4() {
    size_t ni = 9, nj = 6;

    ambit::Dimension dims = {ni, nj};
    Tensor A = build_and_fill("A", dims, a2);
    Tensor B = build_and_fill("B", dims, b2);
    Tensor C = build_and_fill("C", dims, c2);

    double D = A("i,j") * (B("i,j") + C("i,j"));
    double d = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d += a2[i][j] * (b2[i][j] + c2[i][j]);
        }
    }

    return ::std::fabs(D - d);
}

double test_dot_product5() {
    size_t ni = 9, nj = 6, nk = 7;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {nj, nk}, b2);
    Tensor C = build_and_fill("C", {nk, ni}, c2);

    double D = A("i,j") * B("j,k") * C("k,i");
    double d = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                d += a2[i][j] * b2[j][k] * c2[k][i];
            }
        }
    }

    return ::std::fabs(D - d);
}

double test_dot_product6() {
    size_t ni = 9, nj = 6, nk = 7, nl = 5, nm = 8;

    Tensor A = build_and_fill("A", {ni, nj}, a2);
    Tensor B = build_and_fill("B", {nj, nk, nl, nm}, b4);
    Tensor C = build_and_fill("C", {nm, nl, nk, ni}, c4);

    double D = A("i,j") * B("j,k,l,m") * C("m,l,k,i");
    double d = 0.0;

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        d += a2[i][j] * b4[j][k][l][m] * c4[m][l][k][i];
                    }
                }
            }
        }
    }

    return ::std::fabs(D - d);
}

double test_chain_multiply() {
    size_t ni = 9, nj = 6, nk = 4, nl = 5;

    Tensor A = build_and_fill("A", {nl, nj}, a2);
    Tensor B = build_and_fill("B", {ni, nk}, b2);
    Tensor C = build_and_fill("C", {nk, nl}, c2);
    Tensor D = build_and_fill("D", {ni, nj}, d2);

    D("ij") = B("ik") * C("kl") * A("lj");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            d2[i][j] = 0.0;

            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    d2[i][j] += b2[i][k] * c2[k][l] * a2[l][j];
                }
            }
        }
    }

    return difference(D, d2).second;
}

double test_chain_multiply2() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") = A4("ijmn") * B2("km") * C2("ln");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    d4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_chain_multiply3() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") += A4("ijmn") * B2("km") * C2("ln");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_chain_multiply4() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") -= A4("ijmn") * B2("km") * C2("ln");

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] -= a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}
double test_slice2() {
    size_t ni = 7;
    size_t nj = 7;
    size_t nk = 7;
    size_t nl = 7;

    ::std::vector<size_t> dimsC = {ni, nj};
    ::std::vector<size_t> dimsA = {nk, nl};

    Tensor C = build_and_fill("C", dimsC, c2);
    Tensor A = build_and_fill("A", dimsA, a2);

    IndexRange Cinds = {{1L, 5L}, {0L, 4L}};
    IndexRange Ainds = {{0L, 4L}, {2L, 6L}};

    C(Cinds) = A(Ainds);

    for (size_t i = 0; i < Cinds[0][1] - Cinds[0][0]; i++) {
        for (size_t j = 0; j < Cinds[1][1] - Cinds[1][0]; j++) {
            c2[i + Cinds[0][0]][j + Cinds[1][0]] = a2[i + Ainds[0][0]][j + Ainds[1][0]];
        }
    }

    return difference(C, c2).second;
}

double test_Cijkl_equal_Aijab_Bklab_batched() {
    size_t ni = 2;
    size_t nj = 6;
    size_t nk = 8;
    size_t nl = 7;
    size_t na = 5;
    size_t nb = 3;

    Tensor A = build_and_fill("A", {ni, nj, na, nb}, a4);
    Tensor B = build_and_fill("B", {nk, nl, na, nb}, b4);
    Tensor C = build_and_fill("C", {ni, nj, nk, nl}, c4);

    C("ijkl") += batched("j", A("ijab") * B("klab"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    //                    c4[i][j][k][l] = 0;
                    for (size_t a = 0; a < na; ++a) {
                        for (size_t b = 0; b < nb; ++b) {
                            c4[i][j][k][l] += a4[i][j][a][b] * b4[k][l][a][b];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_chain_multiply2_batched() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") = batched("i", A4("ijmn") * B2("km") * C2("ln"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    d4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_chain_multiply3_batched() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") += batched("j", A4("ijmn") * B2("km") * C2("ln"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_chain_multiply4_batched() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") -= batched("kl", A4("ijmn") * B2("km") * C2("ln"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] -= a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_batched() {
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 7;
    size_t nm = 5;
    size_t nn = 5;
    size_t no = 2;
    size_t np = 2;

    ::std::vector<size_t> dimsA = {ni, nj, nm, nn};
    ::std::vector<size_t> dimsB = {nk, nm};
    ::std::vector<size_t> dimsC = {nl, nn};
    ::std::vector<size_t> dimsD = {ni, nj, nk, nl};

    Tensor A4 = build_and_fill("A4", dimsA, a4);
    Tensor B2 = build_and_fill("B2", dimsB, b2);
    Tensor C2 = build_and_fill("C2", dimsC, c2);
    Tensor D4 = build_and_fill("D4", dimsD, d4);

    D4("ijkl") += batched("ijkl", A4("ijmn") * B2("km") * B2("ln"));

    for (size_t i = 0; i < ni; ++i) {
        for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
                for (size_t l = 0; l < nl; ++l) {
                    for (size_t m = 0; m < nm; ++m) {
                        for (size_t n = 0; n < nn; ++n) {
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * b2[l][n];
                        }
                    }
                }
            }
        }
    }

    return difference(D4, d4).second;
}

double test_batched_with_factor() {
    size_t no = 2;

    ::std::vector<size_t> dimsA = {no, no, no, no};
    ::std::vector<size_t> dimsB = {no, no, no, no};
    ::std::vector<size_t> dimsC = {no, no, no, no};

    Tensor A = build_and_fill("A", dimsA, a4);
    Tensor B = build_and_fill("B", dimsB, b4);
    Tensor C = build_and_fill("C", dimsC, c4);

    C("ijrs") = batched("r", 0.5 * A("abrs") * B("ijab"));

    for (size_t i = 0; i < no; ++i) {
        for (size_t j = 0; j < no; ++j) {
            for (size_t r = 0; r < no; ++r) {
                for (size_t s = 0; s < no; ++s) {
                    c4[i][j][r][s] = 0.0;
                    for (size_t a = 0; a < no; ++a) {
                        for (size_t b = 0; b < no; ++b) {
                            c4[i][j][r][s] += 0.5 * a4[a][b][r][s] * b4[i][j][a][b];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

double test_batched_with_factor_permute() {
    size_t no = 4;

    ::std::vector<size_t> dimsA = {no, no, no, no};
    ::std::vector<size_t> dimsB = {no, no, no, no};
    ::std::vector<size_t> dimsC = {no, no, no, no};

    Tensor A = build_and_fill("A", dimsA, a4);
    Tensor B = build_and_fill("B", dimsB, b4);
    Tensor C = build_and_fill("C", dimsC, c4);

    C("ijrs") = batched("r", 0.5 * A("abrs") * B("ijba"));

    for (size_t i = 0; i < no; ++i) {
        for (size_t j = 0; j < no; ++j) {
            for (size_t r = 0; r < no; ++r) {
                for (size_t s = 0; s < no; ++s) {
                    c4[i][j][r][s] = 0.0;
                    for (size_t a = 0; a < no; ++a) {
                        for (size_t b = 0; b < no; ++b) {
                            c4[i][j][r][s] += 0.5 * a4[a][b][r][s] * b4[i][j][b][a];
                        }
                    }
                }
            }
        }
    }

    return difference(C, c4).second;
}

bool test_ambit() {
    srand(time(nullptr));

    //    ambit::initialize(argc, argv);

    auto test_functions = {
        //            Expectation,  test function,  User friendly description
        ::std::make_tuple(kPass, test_wrapper,
                          "8 permutations of C(\"ij\") += A(\"ik\") * B(\"kj\")"),
        ::std::make_tuple(kPass, test_C_equal_2_A, "C(\"ij\") = 2.0 * A(\"ij\")"),
        ::std::make_tuple(kPass, test_C_plus_equal_2_A, "C(\"ij\") += 2.0 * A(\"ij\")"),
        ::std::make_tuple(kPass, test_C_minus_equal_2_A, "C(\"ij\") -= 2.0 * A(\"ij\")"),
        ::std::make_tuple(kPass, test_C_times_equal_2, "C(\"ij\") *= 2.0"),
        ::std::make_tuple(kPass, test_C_divide_equal_2, "C(\"ij\") /= 2.0"),
        ::std::make_tuple(kPass, test_Cij_equal_Aik_Bkj, "C(\"ij\") = A(\"ik\") * B(\"kj\")"),
        ::std::make_tuple(kPass, test_Cij_equal_Aik_Bjk, "C(\"ij\") = A(\"ik\") * B(\"jk\")"),
        ::std::make_tuple(kPass, test_Cij_plus_equal_Aik_Bkj, "C(\"ij\") += A(\"ik\") * B(\"kj\")"),
        ::std::make_tuple(kPass, test_Cij_minus_equal_Aik_Bkj,
                          "C(\"ij\") -= A(\"ik\") * B(\"kj\")"),
        ::std::make_tuple(kPass, test_Cijkl_equal_Aijab_Bklab,
                          "C(\"ijkl\") += A(\"ijab\") * B(\"klab\")"),
        ::std::make_tuple(kPass, test_Cij_equal_Aiabc_Bjabc,
                          "C(\"ij\") += A(\"iabc\") * B(\"jabc\")"),
        ::std::make_tuple(kPass, test_Cikjl_equal_Aijab_Bklab,
                          "C(\"ikjl\") += A(\"ijab\") * B(\"klab\")"),
        ::std::make_tuple(kPass, test_Cij_equal_Aji, "C(\"ij\") = A(\"ji\")"),
        ::std::make_tuple(kPass, test_Cijkl_equal_Akilj, "C(\"ijkl\") = A(\"kilj\")"),
        ::std::make_tuple(kPass, test_Cijkl_equal_Akijl, "C(\"ijkl\") = A(\"kijl\")"),
        ::std::make_tuple(kException, test_Cij_equal_Cij,
                          "C(\"ij\") = C(\"ji\") exception expected"),
        ::std::make_tuple(kPass, test_Cilkj_equal_Aibaj_Bblak,
                          "C(\"ilkj\") += A(\"ibaj\") * B(\"blak\")"),
        ::std::make_tuple(kPass, test_Cljik_equal_Abija_Blbak,
                          "C(\"ljik\") += A(\"bija\") * B(\"lbak\")"),
        ::std::make_tuple(kPass, test_Cij_minus_equal_Aij_Bij,
                          "C(\"ij\") -= A(\"ij\") * B(\"ij\")"),
        ::std::make_tuple(kPass, test_Dij_plus_equal_Aij_Bij_Cij,
                          "D(\"ij\") += A(\"ij\") * B(\"ij\") * C(\"ij\")"),
        ::std::make_tuple(kPass, test_E_abcd_equal_Aijab_Bklcd_C_jl_D_ik,
                          "E(\"abcd\") += A(\"ijab\") * B(\"klcd\") * C(\"jl\") * D(\"ik\")"),
        ::std::make_tuple(kPass, test_Cij_equal_Aij_plus_Bij, "C(\"ij\") = A(\"ij\") + B(\"ij\")"),
        ::std::make_tuple(kPass, test_Dij_equal_Aij_plus_Bij_plus_Cij,
                          "D(\"ij\") = A(\"ij\") + B(\"ij\") + C(\"ij\")"),
        ::std::make_tuple(kPass, test_Cij_equal_Aij_minus_Bij,
                          "C(\"ij\") = A(\"ij\") - 5.0 * B(\"ij\")"),
        ::std::make_tuple(kPass, test_Dij_equal_Aij_minus_Bij_plus_Cij,
                          "D(\"ij\") = A(\"ij\") - B(\"ij\") + 2.0 * C(\"ij\")"),
        ::std::make_tuple(kPass, test_Dij_equal_Aij_times_Bij_plus_Cij,
                          "D(\"ij\") = A(\"ij\") * (2.0 * B(\"ij\") - C(\"ij\"))"),
        ::std::make_tuple(kPass, test_Dij_equal_Bij_plus_Cij_times_Aij,
                          "D(\"ij\") = (2.0 * B(\"ij\") - C(\"ij\")) * A(\"ij\")"),
        ::std::make_tuple(kPass, test_F_equal_D_times_2g_minus_g,
                          "F(\"ij\") = D(\"kl\") * (2.0 * g(\"ijkl\") - g(\"ikjl\"))"),
        ::std::make_tuple(kPass, test_Dij_equal_2_times_Aij_plus_Bij,
                          "C(\"ij\") = 2.0 * (A(\"ij\") - B(\"ij\"))"),
        ::std::make_tuple(kPass, test_Dij_equal_negate_Aij_plus_Bij,
                          "C(\"ij\") = - (A(\"ij\") - B(\"ij\"))"),
        ::std::make_tuple(kPass, test_syev, "Diagonalization (not confirmed)"),
        ::std::make_tuple(kPass, test_geev, "Diagonalization (not confirmed)"),
        ::std::make_tuple(kPass, test_power, "C^(-1/2) (not confirmed)"),
        ::std::make_tuple(kPass, test_dot_product, "double = A(\"ij\")\" * B(\"ij\")"),
        ::std::make_tuple(kException, test_dot_product2,
                          "double = A(\"ij\") * B(\"ik\") exception expected"),
        ::std::make_tuple(kException, test_dot_product3,
                          "double = A(\"ij\") * B(\"ij\") exception expected"),
        ::std::make_tuple(kPass, test_dot_product4,
                          "double D = A(\"i,j\") * (B(\"i,j\") + C(\"i,j\"))"),
        ::std::make_tuple(kPass, test_dot_product5,
                          "double D = A(\"i,j\") * B(\"j,k\") * C(\"k,i\")"),
        ::std::make_tuple(kPass, test_dot_product6,
                          "double D = A(\"i,j\") * B(\"j,k,l,m\") * C(\"m,l,k,i\")"),
        ::std::make_tuple(kPass, test_chain_multiply,
                          "D(\"ij\") = B(\"ik\") * C(\"kl\") * A(\"lj\")"),
        ::std::make_tuple(kPass, test_chain_multiply2,
                          "D4(\"ijkl\") = A4(\"ijmn\") * B2(\"km\") * C2(\"ln\")"),
        ::std::make_tuple(kPass, test_chain_multiply3,
                          "D4(\"ijkl\") += A4(\"ijmn\") * B2(\"km\") * C2(\"ln\")"),
        ::std::make_tuple(kPass, test_chain_multiply4,
                          "D4(\"ijkl\") -= A4(\"ijmn\") * B2(\"km\") * C2(\"ln\")"),
        ::std::make_tuple(kPass, test_slice2, "Slice C2(1:5,0:4) = A2(0:4,2:6)"),
        ::std::make_tuple(kPass, test_Cijkl_equal_Aijab_Bklab_batched,
                          "C(\"ijkl\") += batched(\"j\",A(\"ijab\") * B(\"klab\"))"),
        ::std::make_tuple(kPass, test_chain_multiply2_batched,
                          "D4(\"ijkl\") = batched(\"i\",A4(\"ijmn\") * B2(\"km\") * C2(\"ln\"))"),
        ::std::make_tuple(kPass, test_chain_multiply3_batched,
                          "D4(\"ijkl\") += batched(\"j\",A4(\"ijmn\") * B2(\"km\") * C2(\"ln\"))"),
        ::std::make_tuple(kPass, test_chain_multiply4_batched,
                          "D4(\"ijkl\") -= batched(\"kl\",A4(\"ijmn\") * B2(\"km\") * C2(\"ln\"))"),
        ::std::make_tuple(
            kPass, test_batched,
            "D4(\"ijkl\") += batched(\"ijkl\", A4(\"ijmn\") * B2(\"km\") * B2(\"ln\"))"),
        ::std::make_tuple(kPass, test_batched_with_factor,
                          "C2(\"ijrs\") = batched(\"r\", 0.5 * A(\"abrs\") * B(\"ijab\"))"),
        ::std::make_tuple(kPass, test_batched_with_factor_permute,
                          "C2(\"ijrs\") = batched(\"r\", 0.5 * A(\"abrs\") * B(\"ijba\"))"),
    };

    ::std::vector<::std::tuple<::std::string, TestResult, double>> results;

    outfile->Printf("\n\n==> TEST AMBIT <==\n");

    outfile->Printf("\n %-50s %12s %s", "Description", "Max. error", "Result");
    outfile->Printf("\n %s", ::std::string(83, '-').c_str());

    double max_error = 0.0;

    bool success = true;
    for (auto test_function : test_functions) {
        outfile->Printf("\n %-60s", ::std::get<2>(test_function));
        double result = 0.0;
        TestResult tresult = kPass, report_result = kPass;
        ::std::string exception;
        try {
            result = ::std::get<1>(test_function)();
            max_error = ::std::max(result, max_error);

            // Did the test pass based on returned value?
            tresult = ::std::fabs(result) < zero ? kPass : kFail;
            // Was the tresult the expected result? If so color green else red.
            report_result = tresult == ::std::get<0>(test_function) ? kPass : kFail;
        } catch (::std::exception& e) {
            // was an exception expected?
            tresult = kException;
            report_result = tresult == ::std::get<0>(test_function) ? kPass : kException;

            if (report_result == kException) {
                max_error = ::std::numeric_limits<double>::max();
                exception = e.what();
            }
        }
        outfile->Printf(" %7e", result);
        switch (tresult) {
        case kPass:
            outfile->Printf(" Passed");
            break;
        case kFail:
            outfile->Printf(" Failed");
            break;
        default:
            outfile->Printf(" Exception");
        }

        if (report_result == kException)
            outfile->Printf("\n    Unexpected: %s", exception.c_str());
        if (report_result != kPass)
            success = false;
    }
    outfile->Printf("\n %s", ::std::string(83, '-').c_str());
    outfile->Printf("\n Tests: %s\n", success ? "All passed" : "Some failed");

    Process::environment.globals["AMBIT MAX ERROR"] = max_error;

    return success;
}
}
}
}
