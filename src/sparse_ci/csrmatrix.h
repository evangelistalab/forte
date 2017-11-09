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

#ifndef _csrmatrix_h_
#define _csrmatrix_h_

#include <vector>

namespace psi {
namespace forte {

enum SparseMatrixType { General, Symmetric };

/**
 * A class to store a matrix in compressed sparse row format.
 *
 * see https://en.wikipedia.org/wiki/Sparse_matrix
 */
class CSRMatrix {
  public:
    // Class Constructor and Destructor
    /// Construct an empty CSR matrix
    CSRMatrix();
    /// Construct a CSR matrix of given size
    CSRMatrix(size_t size);

    /// Add a row to the matrix
    void add_row(size_t num_elements, const std::vector<double>& row_value,
                 const std::vector<unsigned int>& row_column_index);

  public:
    // Object Data
    size_t allocated_size_ = 0;
    size_t size_ = 0;
    std::vector<double> value_;
    std::vector<unsigned int> column_index_;
    std::vector<unsigned int> row_offset_ = {0};
};

/**
 * A class to store a matrix in compressed sparse column format.
 *
 * see https://en.wikipedia.org/wiki/Sparse_matrix
 */
class CSCMatrix {
  public:
    // Class Constructor and Destructor
    /// Construct an empty CSC matrix
    CSCMatrix(SparseMatrixType type);
    /// Construct a CSC matrix of given size
    CSCMatrix(SparseMatrixType type, size_t size);

    /// Add a row to the matrix
    void add_column(size_t num_elements, const std::vector<double>& col_value,
                    const std::vector<unsigned int>& col_row_index);

    /// Multiply by a dense vector b = Matrix x a
    void cscmatrix_densevector_multiplication(const std::vector<double>& a, std::vector<double>& b);
    std::vector<unsigned int> column_index_;

  private:
    // Object Data
    SparseMatrixType type_;
    size_t allocated_size_ = 0;
    size_t size_ = 0;
    std::vector<double> value_;
    std::vector<unsigned int> row_index_;
    std::vector<unsigned int> col_offset_ = {0};

    // Class functions
    void general_mv(const std::vector<double>& a, std::vector<double>& b);
    void symmetric_mv(const std::vector<double>& a, std::vector<double>& b);
};
}
} // End Namespaces

#endif // _csrmatrix_h_
