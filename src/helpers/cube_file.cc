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

#include <algorithm>
#include <functional>
#include <fstream>

#include "helpers/string_algorithms.h"

#include "cube_file.h"

namespace forte {

CubeFile::CubeFile(const std::string& filename) : filename_(filename) { load(filename_); }

int CubeFile::natoms() const { return natoms_; }
const std::vector<int>& CubeFile::num() const { return num_; }
const std::vector<double>& CubeFile::min() const { return min_; }
const std::vector<double>& CubeFile::max() const { return max_; }
const std::vector<double>& CubeFile::inc() const { return inc_; }
const std::vector<double>& CubeFile::atom_numbers() const { return atom_numbers_; }
const std::vector<std::tuple<double, double, double>>& CubeFile::atom_coords() const {
    return atom_coords_;
}
const std::vector<double>& CubeFile::data() const { return data_; }

void CubeFile::scale(double factor) {
    for (auto& d : data_) {
        d *= factor;
    }
}

void CubeFile::add(const CubeFile& cf) {
    if (cf.data().size() == data().size()) {
        std::transform(data_.begin(), data_.end(), cf.data().begin(), data_.begin(),
                       std::plus<double>());
    }
}

void CubeFile::pointwise_product(const CubeFile& cf) {
    //    if (cf.data().size() == data().size()) {
    //        std::transform(data_.begin(), data_.end(), cf.data().begin(), data_.begin(),
    //                       std::times<double>());
    //    }
}

void CubeFile::load(std::string filename) {
    std::string line;
    std::vector<std::string> line_split;
    std::ifstream myfile(filename);
    if (myfile.is_open()) {
        // title
        getline(myfile, line);
        title_ = line;

        // comment
        getline(myfile, line);
        comments_ = line;

        // parse comments and set levels

        getline(myfile, line);
        line_split = split_string(line, " ");
        natoms_ = std::stoi(line_split[0]);
        min_ = {std::stod(line_split[1]), std::stod(line_split[2]), std::stod(line_split[3])};

        // info for x coordinate
        for (int n = 0; n < 3; ++n) {
            getline(myfile, line);
            line_split = split_string(line, " ");
            num_.push_back(std::stoi(line_split[0]));
            inc_.push_back(std::stod(line_split[n + 1]));
            max_.push_back(min_[n] + num_[n] * inc_[n]);
        }

        for (int atom = 0; atom < natoms_; ++atom) {
            getline(myfile, line);
            line_split = split_string(line, " ");
            atom_numbers_.push_back(std::stoi(line_split[0]));
            atom_coords_.push_back(
                {std::stod(line_split[2]), std::stod(line_split[3]), std::stod(line_split[4])});
        }

        int ndata = num_[0] * num_[1] * num_[2];
        data_.resize(ndata);

        int remainder = ndata % 6;
        int nbatch = (ndata - remainder) / 6;
        double d[6];
        int batch_start = 0;
        for (int n = 0; n < ndata; n += 6) {
            getline(myfile, line);
            sscanf(line.c_str(), "%lE %lE %lE %lE %lE %lE", &d[0], &d[1], &d[2], &d[3], &d[4],
                   &d[5]);
            int nread = std::min(6, ndata - n);
            for (int k = 0; k < nread; ++k)
                data_[n + k] = d[k];
        }

        //        for (int batch = 0; batch < nbatch; ++batch) {
        //            getline(myfile, line);
        //            sscanf(line.c_str(), "%lE %lE %lE %lE %lE %lE", &d[0], &d[1], &d[2], &d[3],
        //            &d[4],
        //                   &d[5]);
        //            for (int k = 0; k < 6; ++k)
        //                data_[batch * 6 + k] = d[k];
        //        }
        //        if (remainder > 0) {
        //            getline(myfile, line);
        //            sscanf(line.c_str(), "%lE %lE %lE %lE %lE", &d[0], &d[1], &d[2], &d[3],
        //            &d[4]); for (int k = 0; k < remainder; ++k)
        //                data_[nbatch * 6 + k] = d[k];
        //        }

        myfile.close();

        if (data_.size() != num_[0] * num_[1] * num_[2]) {
            throw std::runtime_error(
                "Number of data points is inconsistent with header in Cube file!");
        }
    } else {
        throw std::runtime_error("Unable to open file: " + filename_);
    }
}

} // namespace forte
