/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <fstream>
#include <iostream>
#include <sstream>

#include "ambit/blocked_tensor.h"
#include "helpers/disk_io.h"
#include "sadsrg.h"

using namespace psi;
using namespace ambit;

namespace forte {

std::string SADSRG::write_disk_BT(BlockedTensor& BT, const std::string& name) {
    auto block_labels = BT.block_labels();
    std::vector<std::string> block_file_names;
    block_file_names.reserve(block_labels.size());

    for (const std::string& block : block_labels) {
        size_t nele = BT.block(block).numel();
        std::string filename = filename_prefix_ + "." + name + "." + block + ".bin";
        block_file_names.push_back(block + " " + filename + " " + std::to_string(nele));
        write_disk_vector_double(filename, BT.block(block).data(), true);
    }

    // write master file
    std::ofstream of;
    std::string filename = filename_prefix_ + "." + name + ".master.txt";
    of.open(filename, std::ios::trunc);
    for (const std::string& str : block_file_names) {
        of << str << std::endl;
    }
    of.close();
    return filename;
}

void SADSRG::read_disk_BT(BlockedTensor& BT, const std::string& filename) {
    // read master file info for each block
    std::ifstream infile(filename);
    if (!infile.good()) {
        std::string error = "File " + filename + " does not exist.";
        throw psi::PSIEXCEPTION(error.c_str());
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string block, filename;
        size_t nele;
        iss >> block >> filename >> nele;

        // test if sizes match
        if (nele != BT.block(block).numel()) {
            std::string msg = "Number of elements do NOT match: ";
            msg += BT.name() + "(" + std::to_string(BT.block(block).numel()) + "); ";
            msg += filename + "(" + std::to_string(nele) + ")";
            throw PSIEXCEPTION(msg);
        }

        // read data
        read_disk_vector_double(filename, BT.block(block).data());
    }

    infile.close();
}

void SADSRG::delete_disk_BT(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.good()) {
        std::string error = "File " + filename + " does not exist.";
        throw psi::PSIEXCEPTION(error.c_str());
    }

    // delete every block
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string block, filename_block;
        size_t nele;
        iss >> block >> filename_block >> nele;

        if (remove(filename_block.c_str()) != 0) {
            std::string msg = "Error when deleting " + filename_block;
            perror(msg.c_str());
        }
    }
    infile.close();

    // delete the master file
    if (remove(filename.c_str()) != 0) {
        std::string msg = "Error when deleting " + filename;
        perror(msg.c_str());
    }
}

} // namespace forte
