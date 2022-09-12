/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/writer.h"
#include "psi4/libmints/writer_file_prefix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"

#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"

using namespace psi;

namespace forte {

std::string get_ms_string(double twice_ms) {
    std::string ms_str;
    long twice_ms_long = std::lround(twice_ms);
    if ((twice_ms_long % 2) == 0) {
        ms_str = std::to_string(twice_ms_long / 2);
    } else {
        ms_str.append(std::to_string(twice_ms_long));
        ms_str += "/";
        ms_str += "2";
    }
    return ms_str;
}

py::array_t<double> ambit_to_np(ambit::Tensor t) {
    return py::array_t<double>(t.dims(), &(t.data()[0]));
}

py::array_t<double> vector_to_np(const std::vector<double>& v, const std::vector<size_t>& dims) {
    return py::array_t<double>(dims, &(v.data()[0]));
}

py::array_t<double> vector_to_np(const std::vector<double>& v, const std::vector<int>& dims) {
    return py::array_t<double>(dims, &(v.data()[0]));
}

psi::SharedMatrix tensor_to_matrix(ambit::Tensor t) {
    size_t size1 = t.dim(0);
    size_t size2 = t.dim(1);
    auto M = std::make_shared<psi::Matrix>("M", size1, size2);
    t.iterate([&](const std::vector<size_t>& i, double& value) { M->set(i[0], i[1], value); });
    return M;
}

psi::SharedMatrix tensor_to_matrix(ambit::Tensor t, psi::Dimension dims) {
    if (t.dims().size() != 2) {
        throw std::runtime_error("Unable to convert: Tensor rank is not 2!");
    }

    if (t.dim(0) != t.dim(1)) {
        throw std::runtime_error("Unable to convert: Not square matrix!");
    }

    auto n = static_cast<size_t>(dims.sum());
    if (n != t.dim(0) or n != t.dim(1)) {
        throw std::runtime_error("Unable to convert: Dimension mismatch!");
    }

    auto& t_data = t.data();
    auto M_sym = std::make_shared<psi::Matrix>("M", dims, dims);

    auto nirrep = static_cast<size_t>(dims.n());
    for (size_t h = 0, offset = 0; h < nirrep; ++h) {
        auto size = static_cast<size_t>(dims[h]);
        for (size_t p = 0; p < size; ++p) {
            auto np = p + offset;
            for (size_t q = 0; q < size; ++q) {
                M_sym->set(h, p, q, t_data[np * n + q + offset]);
            }
        }
        offset += dims[h];
    }
    return M_sym;
}

std::pair<double, std::string> to_xb(size_t nele, size_t type_size) {
    if (nele == 0)
        return {0.0, "B"};

    // map the size
    std::map<std::string, double> to_XB;
    to_XB["B"] = 1.0;
    to_XB["KB"] = 1000.0; // use 1000.0 for safety
    to_XB["MB"] = 1000000.0;
    to_XB["GB"] = 1000000000.0;
    to_XB["TB"] = 1000000000000.0;
    to_XB["PB"] = 1000000000000000.0;

    // convert to appropriate unit
    size_t bytes = nele * type_size;
    std::pair<double, std::string> out;
    for (auto& XB : to_XB) {
        double xb = bytes / XB.second;
        if (xb >= 0.9 && xb < 900.0) {
            out = std::make_pair(xb, XB.first);
            break;
        }
    }
    return out;
}

// void view_modified_orbitals(psi::SharedWavefunction wfn, const std::shared_ptr<psi::Matrix>& Ca,
//                             const std::shared_ptr<Vector>& diag_F,
//                             const std::shared_ptr<Vector>& occupation) {
//     std::shared_ptr<MoldenWriter> molden(new MoldenWriter(wfn));
//     std::string filename = get_writer_file_prefix(wfn->molecule()->name()) + ".molden";

//     if (remove(filename.c_str()) == 0) {
//         outfile->Printf("\n  Remove previous molden file named %s.", filename.c_str());
//     }
//     outfile->Printf("\n  Write molden file to %s.", filename.c_str());
//     molden->write(filename, Ca, Ca, diag_F, diag_F, occupation, occupation, true);
// }

// std::pair<std::vector<size_t>, std::vector<size_t>> split_up_tasks(size_t size_of_tasks,
//                                                                    size_t nproc) {
//     size_t mystart = 0;
//     size_t nbatch = 0;
//     std::vector<size_t> mystart_list(nproc, 0);
//     std::vector<size_t> myend_list(nproc, 0);
//     for (size_t me = 0; me < nproc; me++) {
//         mystart = (size_of_tasks / nproc) * me;
//         if (size_of_tasks % nproc > me) {
//             mystart += me;
//             nbatch = mystart + (size_of_tasks / nproc) + 1;
//         } else {
//             mystart += size_of_tasks % nproc;
//             nbatch = mystart + (size_of_tasks / nproc);
//         }
//         mystart_list[me] = mystart;
//         myend_list[me] = nbatch;
//     }
//     std::pair<std::vector<size_t>, std::vector<size_t>> my_lists =
//         std::make_pair(mystart_list, myend_list);

//     return my_lists;
// }

namespace math {
size_t combinations(size_t n, size_t k) {
    if (k > n)
        return 0;
    if (k * 2 > n)
        k = n - k;
    if (k == 0)
        return 1;

    size_t result = n;
    for (size_t i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}
} // namespace math

} // namespace forte
