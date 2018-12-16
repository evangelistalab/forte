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

#include <numeric>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "psi4/psi4-dec.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/writer.h"
#include "psi4/libmints/writer_file_prefix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"

#include "helpers/mo_space_info.h"

using namespace psi;

namespace forte {

// MOSpaceInfo::MOSpaceInfo(psi::Dimension& nmopi) : nirrep_(nmopi.n()), nmopi_(nmopi) {
//     // Add the elementary spaces to the list of composite spaces
//     for (const std::string& es : elementary_spaces_) {
//         composite_spaces_[es] = {es};
//     }
// }
//
// MOSpaceInfo::~MOSpaceInfo() {}
//
// size_t MOSpaceInfo::size(const std::string& space) {
//     size_t s = 0;
//     if (composite_spaces_.count(space) == 0) {
//         std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
//         throw psi::PSIEXCEPTION(msg.c_str());
//     } else {
//         for (const auto& el_space : composite_spaces_[space]) {
//             if (mo_spaces_.count(el_space))
//                 s += mo_spaces_[el_space].first.sum();
//         }
//     }
//     return s;
// }
//
// psi::Dimension MOSpaceInfo::get_dimension(const std::string& space) {
//     psi::Dimension result(nirrep_);
//     if (composite_spaces_.count(space) == 0) {
//         std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
//         throw psi::PSIEXCEPTION(msg.c_str());
//     } else {
//         for (const auto& el_space : composite_spaces_[space]) {
//             if (mo_spaces_.count(el_space))
//                 result += mo_spaces_[el_space].first;
//         }
//     }
//     return result;
// }
//
// /// @return The Slice object for space in a given composite space
// Slice MOSpaceInfo::get_slice(const std::string& space, const std::string& comp_space) {
//     psi::Dimension begin(nirrep_);
//     psi::Dimension end(nirrep_);
//
//     return Slice(begin, end);
// }
//
// std::vector<int> MOSpaceInfo::symmetry(const std::string& space) {
//     psi::Dimension dims = get_dimension(space);
//     std::vector<int> result;
//     for (int h = 0; h < dims.n(); ++h) {
//         for (int i = 0; i < dims[h]; ++i) {
//             result.push_back(h);
//         }
//     }
//     return result;
// }
//
// std::vector<size_t> MOSpaceInfo::get_absolute_mo(const std::string& space) {
//     std::vector<size_t> result;
//     if (composite_spaces_.count(space) == 0) {
//         std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
//         throw psi::PSIEXCEPTION(msg.c_str());
//     } else {
//         for (const auto& el_space : composite_spaces_[space]) {
//             if (mo_spaces_.count(el_space)) {
//                 auto& vec_mo_info = mo_spaces_[el_space].second;
//                 for (auto& mo_info : vec_mo_info) {
//                     result.push_back(std::get<0>(mo_info)); // <- grab the absolute index
//                 }
//             }
//         }
//     }
//     return result;
// }
//
// std::vector<size_t> MOSpaceInfo::get_corr_abs_mo(const std::string& space) {
//     std::vector<size_t> result;
//     if (composite_spaces_.count(space) == 0) {
//         std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
//         throw psi::PSIEXCEPTION(msg.c_str());
//     } else {
//         for (const auto& el_space : composite_spaces_[space]) {
//             if (mo_spaces_.count(el_space)) {
//                 auto& vec_mo_info = mo_spaces_[el_space].second;
//                 for (auto& mo_info : vec_mo_info) {
//                     result.push_back(mo_to_cmo_[std::get<0>(mo_info)]); // <- grab the
//                                                                         // absolute index and
//                                                                         // convert to
//                                                                         // correlated MOs
//                 }
//             }
//         }
//     }
//     return result;
// }
//
// std::vector<std::pair<size_t, size_t>> MOSpaceInfo::get_relative_mo(const std::string& space) {
//     std::vector<std::pair<size_t, size_t>> result;
//     if (composite_spaces_.count(space) == 0) {
//         std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
//         throw psi::PSIEXCEPTION(msg.c_str());
//     } else {
//         for (const auto& el_space : composite_spaces_[space]) {
//             if (mo_spaces_.count(el_space)) {
//                 auto& vec_mo_info = mo_spaces_[el_space].second;
//                 for (auto& mo_info : vec_mo_info) {
//                     result.push_back(std::make_pair(
//                         std::get<1>(mo_info),
//                         std::get<2>(mo_info))); // <- grab the irrep and relative index
//                 }
//             }
//         }
//     }
//     return result;
// }
//
// void MOSpaceInfo::read_options(psi::Options& options) {
//     outfile->Printf("\n\n  ==> MO Space Information <==\n");
//
//     // Read the elementary spaces
//     for (std::string& space : elementary_spaces_) {
//         std::pair<SpaceInfo, bool> result = read_mo_space(space, options);
//         if (result.second) {
//             mo_spaces_[space] = result.first;
//         }
//     }
//
//     // Handle frozen core
//
//     // Count the assigned orbitals
//     psi::Dimension unassigned = nmopi_;
//     for (auto& str_si : mo_spaces_) {
//         unassigned -= str_si.second.first;
//     }
//
//     for (size_t h = 0; h < nirrep_; ++h) {
//         if (unassigned[h] < 0) {
//             outfile->Printf("\n  There is an error in the definition of the "
//                             "orbital spaces.  Total unassigned MOs for irrep "
//                             "%d is %d.",
//                             h, unassigned[h]);
//             exit(1);
//         }
//     }
//
//     // Adjust size of undefined spaces
//     for (std::string space : elementary_spaces_priority_) {
//         // Assign MOs to the undefined space with the highest priority
//         if (not mo_spaces_.count(space)) {
//             std::vector<MOInfo> vec_mo_info;
//             mo_spaces_[space] = std::make_pair(unassigned, vec_mo_info);
//             for (size_t h = 0; h < nirrep_; ++h) {
//                 unassigned[h] = 0;
//             }
//         }
//     }
//     if (unassigned.sum() != 0) {
//         outfile->Printf("\n  There is an error in the definition of the "
//                         "orbital spaces.  There are %d unassigned MOs.",
//                         unassigned.sum());
//         exit(1);
//     }
//
//     // Compute orbital mappings
//     for (size_t h = 0, p_abs = 0; h < nirrep_; ++h) {
//         size_t p_rel = 0;
//         for (std::string space : elementary_spaces_) {
//             size_t n = mo_spaces_[space].first[h];
//             for (size_t q = 0; q < n; ++q) {
//                 mo_spaces_[space].second.push_back(std::make_tuple(p_abs, h, p_rel));
//                 p_abs += 1;
//                 p_rel += 1;
//             }
//         }
//     }
//
//     // Compute the MO to correlated MO mapping
//     std::vector<size_t> vec(nmopi_.sum());
//     std::iota(vec.begin(), vec.end(), 0);
//
//     // Remove the frozen core/virtuals
//     std::vector<int> removed_list;
//     for (MOInfo& mo_info : mo_spaces_["FROZEN_DOCC"].second) {
//         vec.erase(std::remove(vec.begin(), vec.end(), std::get<0>(mo_info)), vec.end());
//         removed_list.push_back(std::get<0>(mo_info));
//     }
//     for (MOInfo& mo_info : mo_spaces_["FROZEN_UOCC"].second) {
//         vec.erase(std::remove(vec.begin(), vec.end(), std::get<0>(mo_info)), vec.end());
//     }
//     //    outfile->Printf("\n Removed orbitals %d", std::get<0>(mo_info));
//
//     mo_to_cmo_.assign(nmopi_.sum(), 1000000000);
//     for (size_t n = 0; n < vec.size(); ++n) {
//         mo_to_cmo_[vec[n]] = n;
//     }
//
//     // Define composite spaces
//
//     // Print the space information
//     size_t label_size = 1;
//     for (std::string space : elementary_spaces_) {
//         label_size = std::max(space.size(), label_size);
//     }
//
//     int banner_width = label_size + 4 + 6 * (nirrep_ + 1);
//     CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
//     outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
//     outfile->Printf("\n    %s", std::string(label_size, ' ').c_str());
//     for (size_t h = 0; h < nirrep_; ++h)
//         outfile->Printf(" %5s", ct.gamma(h).symbol());
//     outfile->Printf("   Sum");
//     outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
//
//     for (std::string space : elementary_spaces_) {
//         psi::Dimension& dim = mo_spaces_[space].first;
//         outfile->Printf("\n    %-*s", label_size, space.c_str());
//         for (size_t h = 0; h < nirrep_; ++h) {
//             outfile->Printf("%6d", dim[h]);
//         }
//         outfile->Printf("%6d", dim.sum());
//     }
//     outfile->Printf("\n    %-*s", label_size, "Total");
//     for (size_t h = 0; h < nirrep_; ++h) {
//         outfile->Printf("%6d", nmopi_[h]);
//     }
//     outfile->Printf("%6d", nmopi_.sum());
//     outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
// }
//
// std::pair<SpaceInfo, bool> MOSpaceInfo::read_mo_space(const std::string& space, psi::Options& options) {
//     bool read = false;
//     psi::Dimension space_dim(nirrep_);
//     std::vector<MOInfo> vec_mo_info;
//     if ((options[space].has_changed()) && (options[space].size() == nirrep_)) {
//         for (size_t h = 0; h < nirrep_; ++h) {
//             space_dim[h] = options[space][h].to_integer();
//         }
//         read = true;
//         outfile->Printf("\n  Read options for space %s", space.c_str());
//     } else {
//         //        outfile->Printf("\n  The size of space \"%s\" (%d) does not
//         //        match the number of irreducible representations (%zu).",
//         //                        space.c_str(),options[space].size(),nirrep_);
//     }
//     SpaceInfo space_info(space_dim, vec_mo_info);
//     return std::make_pair(space_info, read);
// }

void print_h2(const std::string& text, const std::string& left_separator,
              const std::string& right_separator) {
    outfile->Printf("\n\n  %s %s %s\n", left_separator.c_str(), text.c_str(),
                    right_separator.c_str());
}

std::string to_string(const std::vector<std::string>& vec_str, const std::string& sep) {
    if (vec_str.size() == 0)
        return std::string();

    std::string ss;

    std::for_each(vec_str.begin(), vec_str.end() - 1, [&](const std::string& s) { ss += s + sep; });
    ss += vec_str.back();

    return ss;
}

std::string get_ms_string(double twice_ms) {
    std::string ms_str;
    double ms = twice_ms / 2.0;
    if ((static_cast<int>(twice_ms) % 2) == 0) {
        ms_str = std::to_string(static_cast<int>(ms));
    } else {
        int n = static_cast<int>(ms / 0.5);
        ms_str.append(std::to_string(n));
        ms_str += "/";
        ms_str += "2";
    }
    return ms_str;
}

Matrix tensor_to_matrix(ambit::Tensor t, psi::Dimension dims) {
    // Copy the tensor to a plain matrix
    size_t size = dims.sum();
    Matrix M("M", size, size);
    t.iterate([&](const std::vector<size_t>& i, double& value) { M.set(i[0], i[1], value); });

    Matrix M_sym("M", dims, dims);
    size_t offset = 0;
    for (size_t h = 0; h < static_cast<size_t>(dims.n()); ++h) {
        for (size_t p = 0; p < static_cast<size_t>(dims[h]); ++p) {
            for (size_t q = 0; q < static_cast<size_t>(dims[h]); ++q) {
                double value = M.get(p + offset, q + offset);
                M_sym.set(h, p, q, value);
            }
        }
        offset += dims[h];
    }
    return M_sym;
}

psi::SharedMatrix tensor_to_matrix(ambit::Tensor t) {
    size_t size1 = t.dim(0);
    size_t size2 = t.dim(1);
    psi::SharedMatrix M(new Matrix("M", size1, size2));
    t.iterate([&](const std::vector<size_t>& i, double& value) { M->set(i[0], i[1], value); });
    return M;
}

void write_disk_vector_double(const std::string& filename, const std::vector<double>& data) {
    // check if file exists or not
    struct stat buf;
    bool exist = stat(filename.c_str(), &buf) == 0;
    if (exist) {
        std::stringstream error;
        error << "File " << filename << " already exists.";
        throw psi::PSIEXCEPTION(error.str().c_str());
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

std::pair<double, std::string> to_xb(size_t nele, size_t type_size) {
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

void view_modified_orbitals(psi::SharedWavefunction wfn, const std::shared_ptr<psi::Matrix>& Ca,
                            const std::shared_ptr<Vector>& diag_F,
                            const std::shared_ptr<Vector>& occupation) {
    std::shared_ptr<MoldenWriter> molden(new MoldenWriter(wfn));
    std::string filename = get_writer_file_prefix(wfn->molecule()->name()) + ".molden";

    if (remove(filename.c_str()) == 0) {
        outfile->Printf("\n  Remove previous molden file named %s.", filename.c_str());
    }
    outfile->Printf("\n  Write molden file to %s.", filename.c_str());
    /* PORTTODO: re-enable this block
    molden->write(filename, Ca, Ca, diag_F, diag_F, occupation, occupation);
    */
}
std::pair<std::vector<size_t>, std::vector<size_t>> split_up_tasks(size_t size_of_tasks, size_t nproc) {
    size_t mystart = 0;
    size_t nbatch = 0;
    std::vector<size_t> mystart_list(nproc, 0);
    std::vector<size_t> myend_list(nproc, 0);
    for (size_t me = 0; me < nproc; me++) {
        mystart = (size_of_tasks / nproc) * me;
        if (size_of_tasks % nproc > me) {
            mystart += me;
            nbatch = mystart + (size_of_tasks / nproc) + 1;
        } else {
            mystart += size_of_tasks % nproc;
            nbatch = mystart + (size_of_tasks / nproc);
        }
        mystart_list[me] = mystart;
        myend_list[me] = nbatch;
    }
    std::pair<std::vector<size_t>, std::vector<size_t>> my_lists =
        std::make_pair(mystart_list, myend_list);

    return my_lists;
}

} // namespace forte

