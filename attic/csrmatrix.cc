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
#include "psi4/libpsi4util/PsiOutStream.h"

#include "csrmatrix.h"

using namespace psi;

namespace psi {
namespace forte {

CSRMatrix::CSRMatrix(size_t size) : allocated_size_(size) {
    value_.reserve(size);
    column_index_.reserve(size);
}

void CSRMatrix::add_row(size_t num_elements, const std::vector<double>& row_value,
                        const std::vector<unsigned int>& row_column_index) {
    size_ += num_elements;
    value_.insert(value_.end(), row_value.begin(), row_value.begin() + num_elements);
    column_index_.insert(column_index_.end(), row_column_index.begin(),
                         row_column_index.begin() + num_elements);
    row_offset_.push_back(size_);
}

CSCMatrix::CSCMatrix(SparseMatrixType type) : type_(type) {}

CSCMatrix::CSCMatrix(SparseMatrixType type, size_t size) : allocated_size_(size) {
    value_.reserve(size);
    column_index_.reserve(size);
}

void CSCMatrix::add_column(size_t num_elements, const std::vector<double>& col_value,
                           const std::vector<unsigned int>& col_row_index) {
    size_ += num_elements;
    value_.insert(value_.end(), col_value.begin(), col_value.begin() + num_elements);
    row_index_.insert(row_index_.end(), col_row_index.begin(),
                      col_row_index.begin() + num_elements);
    col_offset_.push_back(size_);
}

void CSCMatrix::cscmatrix_densevector_multiplication(const std::vector<double>& a,
                                                     std::vector<double>& b) {
    if (type_ == General) {
        general_mv(a, b);
    } else if (type_ == Symmetric) {
        symmetric_mv(a, b);
    }
}

void CSCMatrix::general_mv(const std::vector<double>& a, std::vector<double>& b) {
    size_t num_col = col_offset_.size() - 1;
    for (size_t j = 0; j < num_col; j++) {
        const double a_j = a[j];
        const size_t min_i_idx = col_offset_[j];
        const size_t max_i_idx = col_offset_[j + 1];
        for (unsigned int i_idx = min_i_idx; i_idx < max_i_idx; ++i_idx) {
            const size_t i = row_index_[i_idx];
            b[i] += value_[i_idx] * a_j;
        }
    }
}


void CSCMatrix::symmetric_mv(const std::vector<double>& a, std::vector<double>& b) {
    //    size_t num_col = col_offset_.size() - 1;
    //    for (size_t j = 0; j < num_col; j++) {
    //        const double a_j = a[j];
    //        const size_t min_ii = col_offset_[j];
    //        const size_t max_ii = col_offset_[j + 1];
    //        for (unsigned int ii = min_ii; ii < max_ii; ++ii) {
    //            const size_t i = row_index_[ii];
    //            b[i] += value_[ii] * a_j;
    //        }
    //    }
}
}
} // end namespace
