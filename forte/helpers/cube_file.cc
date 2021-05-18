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
#include <regex>
#include <cmath>
#include <numeric>

#include "helpers/string_algorithms.h"

#include "cube_file.h"

namespace forte {

CubeFile::CubeFile(const std::string& filename) { load(filename); }

int CubeFile::natoms() const { return natoms_; }
const std::vector<int>& CubeFile::num() const { return num_; }
const std::vector<double>& CubeFile::min() const { return min_; }
const std::vector<double>& CubeFile::max() const { return max_; }
const std::vector<double>& CubeFile::inc() const { return inc_; }
const std::vector<int>& CubeFile::atom_numbers() const { return atom_numbers_; }
const std::vector<std::tuple<double, double, double>>& CubeFile::atom_coords() const {
    return atom_coords_;
}
const std::vector<double>& CubeFile::data() const { return data_; }

void CubeFile::zero() { std::fill(data_.begin(), data_.end(), 0.0); }

void CubeFile::scale(double factor) {
    for (auto& d : data_) {
        d *= factor;
    }
}

void CubeFile::add(const CubeFile& other, double factor) {
    if (other.data().size() == data().size()) {
        std::transform(data_.begin(), data_.end(), other.data().begin(), data_.begin(),
                       [&](double i, double j) { return i + factor * j; });
    }
}

void CubeFile::pointwise_product(const CubeFile& other) {
    if (other.data().size() == data().size()) {
        std::transform(data_.begin(), data_.end(), other.data().begin(), data_.begin(),
                       [](double i, double j) { return i * j; });
    }
}

std::pair<double, double> CubeFile::compute_levels(std::string type, double fraction) const {
    std::vector<double> sorted_data(data_);
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](double i, double j) { return std::fabs(i) > std::fabs(j); });
    double power = 2.;
    if (type == "density") {
        power = 1.;
    } else if (type == "mo") {
        power = 2.;
    }

    double neg_level = 0.0;
    double pos_level = 0.0;
    double sum = std::accumulate(data_.begin(), data_.end(), 0.0,
                                 [&](double i, double j) { return i + std::pow(j, power); });
    double partial_sum = 0;
    for (size_t n = 0, maxn = sorted_data.size(); n < maxn; ++n) {
        partial_sum += std::pow(sorted_data[n], power);
        if (partial_sum / sum < fraction) {
            if (sorted_data[n] < 0.0) {
                neg_level = sorted_data[n];
            } else {
                pos_level = sorted_data[n];
            }
        } else {
            break;
        }
    }
    return std::make_pair(pos_level, neg_level);
}

void CubeFile::load(std::string filename) {
    std::string line;
    std::vector<std::string> line_split;
    std::ifstream file(filename);
    if (file.is_open()) {
        // 1. title
        getline(file, line);
        title_ = line;

        // 2. comment
        getline(file, line);
        comments_ = line;

        // 3. number of atoms plus origin of grid
        getline(file, line);
        line_split = split_string(line, " ");
        natoms_ = std::stoi(line_split[0]);
        min_ = {std::stod(line_split[1]), std::stod(line_split[2]), std::stod(line_split[3])};

        // 4. number of points along axis, displacement along x,y,z
        for (int n = 0; n < 3; ++n) {
            getline(file, line);
            line_split = split_string(line, " ");
            num_.push_back(std::stoi(line_split[0]));
            inc_.push_back(std::stod(line_split[n + 1]));
            max_.push_back(min_[n] + num_[n] * inc_[n]);
        }
        // 5. atoms of molecule (Z, Q?, x, y, z)
        for (int atom = 0; atom < natoms_; ++atom) {
            getline(file, line);
            line_split = split_string(line, " ");
            atom_numbers_.push_back(std::stoi(line_split[0]));
            atom_coords_.push_back(
                {std::stod(line_split[2]), std::stod(line_split[3]), std::stod(line_split[4])});
        }
        // 6. Data, striped (x, y, z)
        int ndata = num_[0] * num_[1] * num_[2];
        data_.resize(ndata);

        double d[6];
        for (int n = 0; n < ndata; n += 6) {
            getline(file, line);
            sscanf(line.c_str(), "%lE %lE %lE %lE %lE %lE", &d[0], &d[1], &d[2], &d[3], &d[4],
                   &d[5]);
            int nread = std::min(6, ndata - n);
            for (int k = 0; k < nread; ++k)
                data_[n + k] = d[k];
        }

        file.close();

        if (data_.size() != num_[0] * num_[1] * num_[2]) {
            throw std::runtime_error(
                "Number of data points is inconsistent with header in Cube file!");
        }
    } else {
        throw std::runtime_error("Unable to open input file: " + filename);
    }
}

void CubeFile::save(std::string filename) const {
    FILE* fh = fopen(filename.c_str(), "w");
    // 1. title
    fprintf(fh, "%s\n", title_.c_str());
    // 2. comment
    fprintf(fh, "%s\n", comments_.c_str());
    // 3. number of atoms plus origin of grid
    fprintf(fh, "%6d %10.6f %10.6f %10.6f\n", natoms_, min_[0], min_[1], min_[2]);
    // 4. number of points along axis, displacement along x,y,z
    fprintf(fh, "%6d %10.6f %10.6f %10.6f\n", num_[0], inc_[0], 0.0, 0.0);
    fprintf(fh, "%6d %10.6f %10.6f %10.6f\n", num_[1], 0.0, inc_[1], 0.0);
    fprintf(fh, "%6d %10.6f %10.6f %10.6f\n", num_[2], 0.0, 0.0, inc_[2]);
    // 5. atoms of molecule (Z, Q?, x, y, z)
    for (int A = 0; A < natoms_; A++) {
        fprintf(fh, "%3d %10.6f %10.6f %10.6f %10.6f\n", atom_numbers_[A], 0.0,
                std::get<0>(atom_coords_[A]), std::get<1>(atom_coords_[A]),
                std::get<2>(atom_coords_[A]));
    }

    // Data, striped (x, y, z)
    for (size_t ind = 0, npoints = data_.size(); ind < npoints; ind++) {
        fprintf(fh, "%12.5E ", data_[ind]);
        if (ind % 6 == 5)
            fprintf(fh, "\n");
    }

    fclose(fh);
}

} // namespace forte
