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

#ifndef _disk_io_h_
#define _disk_io_h_

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

namespace forte {

/**
 * @brief Save a vector of double to file
 * @param filename The file name
 * @param data The data to be dumped
 * @param overwrite Overwrite if the file exists
 */
void write_disk_vector_double(const std::string& filename, const std::vector<double>& data,
                              bool overwrite = false);

/**
 * @brief Read a vector of double from file
 * @param filename The file name
 * @param data The data to be read
 */
void read_disk_vector_double(const std::string& filename, std::vector<double>& data);

/**
 * @brief Save a BlockedTensor to file
 * @param BT The BlockedTensor to be dumped to files
 * @param name The abbreviated name of the BlockedTensor BT
 * @return The master file name (with absolute path) that handles all blocks
 */
std::string write_disk_BT(ambit::BlockedTensor& BT, const std::string& name,
                          const std::string& file_prefix);

/**
 * @brief Read a BlockedTensor from file
 * @param BT The BlockedTensor to be filled
 * @param filename The master file name (with absolute path) that stores file names for all blocks
 */
void read_disk_BT(ambit::BlockedTensor& BT, const std::string& filename);

/**
 * @brief Delete all files written by write_disk_BT
 * @param filename The master file name (with absolute path) that stores file names for all blocks
 */
void delete_disk_BT(const std::string& filename);

} // namespace forte

#endif // _disk_io_h_
