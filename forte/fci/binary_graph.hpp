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

#ifndef _binary_graph_hpp_
#define _binary_graph_hpp_

/*
 *  binary_graph.hpp
 *  binary_graph
 *
 *  Created by Francesco Evangelista on 3/6/09.
 *
 */

#include <iostream>
#include <vector>

#include "sparse_ci/determinant.h"

namespace forte {

/**
 * Compute the address of a string of bits.  Strings are grouped by
 * symmetry.
 */
class BinaryGraph {
  public:
    /*
     * Constructor which takes a vector with size of each irrep
     * @param n lenght of the string
     * @param k number of 1s
     * @param symm symmetry of each orbital
     */

    /// @brief Construct a BinaryGraph object to address strings of bits
    /// @param n the number of bits in the string
    /// @param k the number of 1s in the string
    /// @param irrep_size a vector with the size of each irrep
    BinaryGraph(int n, int k, std::vector<int>& irrep_size)
        : nbits_(n), nones_(k), nirrep_(irrep_size.size()), strpi_(nirrep_, 0),
          offset_(nirrep_, 0) {
        for (int h = 0; h < static_cast<int>(irrep_size.size()); ++h) {
            for (int i = 0; i < irrep_size[h]; ++i) {
                symmetry.push_back(h);
            }
        }
        startup();
        generate_weights();
    }

    /// @brief Return the address of a string
    size_t rel_add(const String& string) {
        size_t add = 0;
        int k = 0; // number of 1s
        int h = 0; // irrep of the string
        for (int n = 0; (k < nones_) and (n < nbits_); ++n) {
            if (string[n]) {
                ++k;
                h ^= symmetry[n];
                add += weight0[index(n, h, k)];
            }
        }
        return add;
    }

    /// @brief Return the symmetry of a string
    int sym(const String& string) { return string.symmetry(symmetry); }

    /// @brief Return the number of strings
    size_t nstr() {
        size_t sum = 0;
        for (int h = 0; h < nirrep_; ++h) {
            sum += strpi_[h];
        }
        return sum;
    }

    /// @brief Return the number of strings with a given symmetry
    size_t strpi(int h) const { return strpi_[h]; }
    /// @brief Return the number of bits in the string
    int nbits() const { return nbits_; }
    /// @brief Return the number of 1s in the string
    int nones() const { return nones_; }

  private:
    // Helper function to convert 3D index to 1D index
    size_t index(int n, int h, int k) { return n * nirrep_ * (nones_ + 1) + h * (nones_ + 1) + k; }

    void startup() {
        if ((nbits_ != 0) and (nones_ > 0)) {

            // Calculate the size of the weight tensors
            const size_t size = nbits_ * nirrep_ * (nones_ + 1);

            // Allocate the weight tensors
            weight0.resize(size, 0);
            weight1.resize(size, 0);

            // Assign values to the weight tensors
            for (int n = 0; n < nbits_; ++n) {
                for (int h = 0; h < nirrep_; ++h) {
                    for (int k = 0; k < nones_ + 1; ++k) {
                        weight0[index(n, h, k)] = 0;
                        weight1[index(n, h, k)] = 0;
                    }
                }
            }
        }
    }

    void generate_weights() {
        if ((nbits_ != 0) and (nones_ > 0)) {
            // Generate weights
            weight0[index(0, 0, 0)] = 1;
            weight1[index(0, symmetry[0], 1)] = 1;
            for (int n = 1; n < nbits_; ++n) {
                // 0 path does not change symmetry
                for (int h = 0; h < nirrep_; ++h) {
                    // Grab the number of paths inherited from the 0 vertex
                    for (int k = 0; k < nones_ + 1; ++k)
                        weight0[index(n, h, k)] += weight0[index(n - 1, h, k)];
                    // Grab the number of paths inherited from the 1 vertex
                    for (int k = 0; k < nones_ + 1; ++k)
                        weight0[index(n, h, k)] += weight1[index(n - 1, h, k)];
                }

                // 1 path changes symmetry by symmetry[n]
                for (int h = 0; h < nirrep_; ++h) {
                    // Grab the number of paths inherited from the 0 vertex
                    for (int k = 0; k < nones_; ++k)
                        weight1[index(n, h ^ symmetry[n], k + 1)] += weight0[index(n - 1, h, k)];
                    // Grab the number of paths inherited from the 1 vertex
                    for (int k = 0; k < nones_; ++k)
                        weight1[index(n, h ^ symmetry[n], k + 1)] += weight1[index(n - 1, h, k)];
                }
            }

            // Generate strings per irrep
            for (int h = 0; h < nirrep_; ++h) {
                strpi_[h] =
                    weight0[index(nbits_ - 1, h, nones_)] + weight1[index(nbits_ - 1, h, nones_)];
            }

            // Generate the offset
            offset_[0] = 0;
            for (int h = 1; h < nirrep_; ++h) {
                offset_[h] = offset_[h - 1] + weight0[index(nbits_ - 1, h - 1, nones_)] +
                             weight1[index(nbits_ - 1, h - 1, nones_)];
            }
        } else {
            for (int h = 0; h < nirrep_; ++h) {
                strpi_[h] = 0;
            }
            strpi_[0] = 1;

            // Generate the offset
            for (int h = 0; h < nirrep_; ++h) {
                offset_[h] = 0;
            }
        }
    }

    int nbits_;                  // number of digits
    int nones_;                  // number of 1s
    int nirrep_;                 // number of irreps
    std::vector<int> symmetry;   // symmetry of each bit
    std::vector<size_t> strpi_;  // strings per irrep
    std::vector<size_t> offset_; // irrep offset
    std::vector<size_t> weight0; // weights of 0 vertices
    std::vector<size_t> weight1; // weights of 1 vertices
};

} // namespace forte

#endif // _binary_graph_hpp_
