/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <numeric>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <iomanip>

#include "psi4/psi4-dec.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/writer.h"
#include "psi4/libmints/writer_file_prefix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"

#include "helpers/disk_io.h"

using namespace psi;

namespace forte {

void write_disk_vector_double(const std::string& filename, const std::vector<double>& data,
                              bool overwrite) {
    // check if file exists or not
    struct stat buf;
    if (stat(filename.c_str(), &buf) == 0) {
        if (overwrite) {
            // delete the file
            if (remove(filename.c_str()) != 0) {
                std::string msg = "Error when deleting " + filename;
                perror(msg.c_str());
            }
        } else {
            std::string error = "File " + filename + " already exists.";
            throw psi::PSIEXCEPTION(error.c_str());
        }
    }

    // write data to file
    // for convenience, write the size to file as well
    // note that &vector<T>[0] is a pointer to type T.
    std::ofstream out(filename.c_str(), std::ios_base::binary);
    size_t data_size = data.size();
    out.write(reinterpret_cast<char*>(&data_size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&data[0]), data_size * sizeof(double));

    out.close();
}

void read_disk_vector_double(const std::string& filename, std::vector<double>& data) {
    // check if file exists or not
    std::ifstream in(filename.c_str(), std::ios_base::binary);
    if (!in.good()) {
        std::stringstream error;
        error << "File " << filename << " does not exist.";
        throw psi::PSIEXCEPTION(error.str().c_str());
    }

    // read file to data
    size_t data_size;
    in.read(reinterpret_cast<char*>(&data_size), sizeof(size_t));
    data.resize(data_size);
    in.read(reinterpret_cast<char*>(&data[0]), data_size * sizeof(double));

    in.close();
}

void write_psi_vector(const std::string& filename, psi::Vector vec, psi::Dimension padding,
                      bool overwrite) {
    // check if file exists or not
    struct stat buf;
    if (stat(filename.c_str(), &buf) == 0) {
        if (overwrite) {
            // delete the file
            if (remove(filename.c_str()) != 0) {
                std::string msg = "Error when deleting " + filename;
                perror(msg.c_str());
            }
        } else {
            std::string error = "File " + filename + " already exists.";
            throw psi::PSIEXCEPTION(error.c_str());
        }
    }

    std::ofstream file(filename);
    file << "# Irrep  Index  Value";
    for (int h = 0, nirrep = vec.nirrep(); h < nirrep; ++h) {
        for (int i = 0, limit = vec.dim(h), shift = padding[h]; i < limit; ++i) {
            file << '\n' << h << "  " << std::setw(6) << i + shift;
            file << std::scientific << std::setprecision(12);
            file << std::setw(20) << vec.get(h, i);
        }
    }
    file.close();
}

void write_psi_matrix(const std::string& filename, const psi::Matrix& mat, bool overwrite) {
    int nirrep = mat.nirrep();
    int symmetry = mat.symmetry();
    size_t n = 0;
    for (int h = 0; h < nirrep; ++h) {
        n += mat.rowspi(h) * mat.colspi(h ^ symmetry);
    }
    // copy the data from the matrix to the vector
    std::vector<double> data(n);
    size_t k = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0, maxi = mat.rowspi(h); i < maxi; ++i) {
            for (int j = 0, maxj = mat.colspi(h ^ symmetry); j < maxj; ++j) {
                data[k] = mat.get(h, i, j);
                ++k;
            }
        }
    }
    write_disk_vector_double(filename, data, overwrite);
}

void read_psi_matrix(const std::string& filename, psi::Matrix& mat) {
    int nirrep = mat.nirrep();
    int symmetry = mat.symmetry();
    size_t n = 0;
    for (int h = 0; h < nirrep; ++h) {
        n += mat.rowspi(h) * mat.colspi(h ^ symmetry);
    }
    // copy the data from the matrix to the vector
    std::vector<double> data(n);
    read_disk_vector_double(filename, data);
    size_t k = 0;
    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0, maxi = mat.rowspi(h); i < maxi; ++i) {
            for (int j = 0, maxj = mat.colspi(h ^ symmetry); j < maxj; ++j) {
                mat.set(h, i, j, data[k]);
                ++k;
            }
        }
    }
}

void dump_occupations(const std::string& filename,
                      std::unordered_map<std::string, psi::Dimension> occ_map) {
    int nirrep = -1;
    std::vector<std::string> spaces;
    for (const auto& [space_name, dim] : occ_map) {
        int n = static_cast<int>(dim.n());
        if (nirrep == -1) {
            nirrep = n;
        } else {
            if (nirrep != n)
                throw std::runtime_error("Inconsistent number of irreps!");
        }
        spaces.push_back(space_name);
    }
    std::ofstream wfile(filename + ".json");
    if (wfile.is_open()) {
        wfile << "{";
        for (int i = 0, n = spaces.size(); i < n; ++i) {
            wfile << "\n    \"" << spaces[i] << "\": [";
            for (int h = 0; h < nirrep; ++h) {
                wfile << occ_map.at(spaces[i])[h];
                if (h < (nirrep - 1))
                    wfile << ",";
            }
            wfile << "]";
            if (i < (n - 1))
                wfile << ",";
        }
        wfile << "\n}";
    } else {
        throw std::runtime_error("Unable to open file for dump occupation numbers!");
    }
}

// std::string write_disk_BT(ambit::BlockedTensor& BT, const std::string& name,
//                          const std::string& file_prefix) {
//    auto block_labels = BT.block_labels();
//    std::vector<std::string> block_file_names;
//    block_file_names.reserve(block_labels.size());

//    for (const std::string& block : block_labels) {
//        size_t nele = BT.block(block).numel();

//        // need to deal with case insensitivity
//        std::string block_lowercase, spin;
//        for (const char& i : block) {
//            if (isupper(i)) {
//                block_lowercase += tolower(i);
//                spin += 'b';
//            } else {
//                block_lowercase += i;
//                spin += 'a';
//            }
//        }

//        std::string filename = file_prefix;
//        for (const std::string& s : {name, block_lowercase, spin, std::string("bin")}) {
//            filename += "." + s;
//        }
//        block_file_names.push_back(block + " " + filename + " " + std::to_string(nele));

//        write_disk_vector_double(filename, BT.block(block).data(), true);
//    }

//    // write master file
//    std::ofstream of;
//    std::string file_path = file_prefix + "." + name + ".master.txt";
//    of.open(file_path, std::ios::trunc);
//    for (const std::string& str : block_file_names) {
//        of << str << std::endl;
//    }
//    of.close();
//    return file_path;
//}

// void read_disk_BT(ambit::BlockedTensor& BT, const std::string& filename) {
//    // read master file info for each block
//    std::ifstream infile(filename);
//    if (!infile.good()) {
//        std::string error = "File " + filename + " does not exist.";
//        throw psi::PSIEXCEPTION(error.c_str());
//    }

//    std::string line;
//    while (std::getline(infile, line)) {
//        std::istringstream iss(line);
//        std::string block, filename;
//        size_t nele;
//        iss >> block >> filename >> nele;

//        // test if sizes match
//        if (nele != BT.block(block).numel()) {
//            std::string msg = "Number of elements do NOT match: ";
//            msg += BT.name() + "(" + std::to_string(BT.block(block).numel()) + "); ";
//            msg += filename + "(" + std::to_string(nele) + ")";
//            throw PSIEXCEPTION(msg);
//        }

//        // read data
//        read_disk_vector_double(filename, BT.block(block).data());
//    }

//    infile.close();
//}

// void delete_disk_BT(const std::string& filename) {
//    std::ifstream infile(filename);
//    if (!infile.good()) {
//        std::string error = "File " + filename + " does not exist.";
//        throw psi::PSIEXCEPTION(error.c_str());
//    }

//    // delete every block
//    std::string line;
//    while (std::getline(infile, line)) {
//        std::istringstream iss(line);
//        std::string block, filename_block;
//        size_t nele;
//        iss >> block >> filename_block >> nele;

//        if (remove(filename_block.c_str()) != 0) {
//            std::string msg = "Error when deleting " + filename_block;
//            perror(msg.c_str());
//        }
//    }
//    infile.close();

//    // delete the master file
//    if (remove(filename.c_str()) != 0) {
//        std::string msg = "Error when deleting " + filename;
//        perror(msg.c_str());
//    }
//}
} // namespace forte
