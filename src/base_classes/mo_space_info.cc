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

#include <algorithm>
#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"

using namespace psi;

namespace forte {

MOSpaceInfo::MOSpaceInfo(const psi::Dimension& nmopi, const std::string& point_group)
    : symmetry_(point_group), nirrep_(nmopi.n()), nmopi_(nmopi) {
    // Add the elementary spaces to the list of composite spaces
    for (const std::string& es : elementary_spaces_) {
        composite_spaces_[es] = {es};
    }
}

size_t MOSpaceInfo::size(const std::string& space) {
    size_t s = 0;
    if (composite_spaces_.count(space) == 0) {
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    } else {
        for (const auto& el_space : composite_spaces_[space]) {
            if (mo_spaces_.count(el_space))
                s += mo_spaces_[el_space].first.sum();
        }
    }
    return s;
}

psi::Dimension MOSpaceInfo::dimension(const std::string& space) {
    psi::Dimension result(nirrep_);
    if (composite_spaces_.count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::dimension - composite space " + space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    } else {
        for (const auto& el_space : composite_spaces_[space]) {
            if (mo_spaces_.count(el_space))
                result += mo_spaces_[el_space].first;
        }
    }
    return result;
}

std::vector<int> MOSpaceInfo::symmetry(const std::string& space) {
    psi::Dimension dims = dimension(space);
    std::vector<int> result;
    for (int h = 0; h < dims.n(); ++h) {
        for (int i = 0; i < dims[h]; ++i) {
            result.push_back(h);
        }
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::absolute_mo(const std::string& space) {
    std::vector<size_t> result;
    if (composite_spaces_.count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::absolute_mo - composite space " + space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    } else {
        std::vector<std::vector<size_t>> mo_list(nirrep_);
        for (const auto& el_space : composite_spaces_[space]) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info) {
                    size_t h = std::get<1>(mo_info);            // <- the orbital irrep
                    mo_list[h].push_back(std::get<0>(mo_info)); // <- grab the absolute index
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::corr_absolute_mo(const std::string& space) {
    std::vector<size_t> result;
    if (composite_spaces_.count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::corr_absolute_mo - composite space " + space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    } else {
        std::vector<std::vector<size_t>> mo_list(nirrep_);
        // Loop over all the spaces
        for (const auto& el_space : composite_spaces_[space]) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info) {
                    size_t h = std::get<1>(mo_info); // <- the orbital irrep
                    mo_list[h].push_back(
                        mo_to_cmo_[std::get<0>(mo_info)]); // <- grab the absolute index
                                                           // and convert to correlated MOs
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

std::vector<std::pair<size_t, size_t>> MOSpaceInfo::relative_mo(const std::string& space) {
    std::vector<std::pair<size_t, size_t>> result;
    if (composite_spaces_.count(space) == 0) {
        std::string msg =
            "\n  MOSpaceInfo::get_relative_mo - composite space " + space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    } else {
        std::vector<std::vector<std::pair<size_t, size_t>>> mo_list(nirrep_);

        for (const auto& el_space : composite_spaces_[space]) {
            if (mo_spaces_.count(el_space)) {
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info) {
                    size_t h = std::get<1>(mo_info); // <- the orbital irrep
                    mo_list[h].push_back(std::make_pair(
                        std::get<1>(mo_info),
                        std::get<2>(mo_info))); // <- grab the irrep and relative index
                }
            }
        }
        for (const auto& irrep : mo_list) {
            for (const auto& p : irrep) {
                result.push_back(p);
            }
        }
    }
    return result;
}

std::vector<std::pair<size_t, size_t>> MOSpaceInfo::get_relative_mo(const std::string& space) {
    return relative_mo(space);
}

std::vector<size_t> MOSpaceInfo::pos_in_space(const std::string& space,
                                              const std::string& composite_space) {
    std::vector<size_t> result;
    if (composite_spaces_.count(space) * composite_spaces_.count(composite_space) == 0) {
        std::string msg = "\n  MOSpaceInfo::pos_in_space - space " + space +
                          " or composite space " + composite_space + " is not defined.";
        throw psi::PSIEXCEPTION(msg.c_str());
    }

    // make sure that space is contained in composite_space
    for (auto s : composite_spaces_[space]) {
        auto it = find(composite_spaces_[composite_space].begin(),
                       composite_spaces_[composite_space].end(), s);
        if (it == composite_spaces_[composite_space].end()) {
            std::string msg = "\n  MOSpaceInfo::pos_in_space - space " + s +
                              " is not contained in composite space " + composite_space + " .";
            throw psi::PSIEXCEPTION(msg.c_str());
        }
    }

    auto abs_space = absolute_mo(space);
    auto abs_composite_space = absolute_mo(composite_space);
    std::unordered_map<size_t, size_t> composite_space_hash;
    size_t k = 0;
    for (size_t p : abs_composite_space) {
        composite_space_hash[p] = k;
        k += 1;
    }
    for (size_t p : abs_space) {
        result.push_back(composite_space_hash[p]);
    }
    return result;
}

void MOSpaceInfo::read_options(std::shared_ptr<ForteOptions> options) {
    // Read the elementary spaces
    for (const std::string& space : elementary_spaces_) {
        std::pair<SpaceInfo, bool> result = read_mo_space(space, options);
        if (result.second) {
            mo_spaces_[space] = result.first;
        }
    }
    for (auto& space_list : composite_spaces_) {
        const auto& space = space_list.first;
        std::pair<SpaceInfo, bool> result = read_mo_space(space, options);
        if (result.second) {
            mo_spaces_[space_list.second[0]] = result.first;
        }
    }
}

void MOSpaceInfo::read_from_map(std::map<std::string, std::vector<size_t>>& mo_space_map) {
    // Read the elementary spaces
    for (std::string& space : elementary_spaces_) {
        std::pair<SpaceInfo, bool> result = read_mo_space_from_map(space, mo_space_map);
        if (result.second) {
            mo_spaces_[space] = result.first;
        }
    }
    for (auto& space_list : composite_spaces_) {
        const auto& space = space_list.first;
        std::pair<SpaceInfo, bool> result = read_mo_space_from_map(space, mo_space_map);
        if (result.second) {
            mo_spaces_[space_list.second[0]] = result.first;
        }
    }
}

void MOSpaceInfo::set_reorder(const std::vector<size_t>& reorder) { reorder_ = reorder; }

void MOSpaceInfo::compute_space_info() {
    outfile->Printf("\n\n  ==> MO Space Information <==\n");

    // Handle frozen core

    // Count the assigned orbitals
    psi::Dimension unassigned = nmopi_;
    for (auto& str_si : mo_spaces_) {
        unassigned -= str_si.second.first;
    }

    for (size_t h = 0; h < nirrep_; ++h) {
        if (unassigned[h] < 0) {
            outfile->Printf("\n  There is an error in the definition of the "
                            "orbital spaces.  Total unassigned MOs for irrep "
                            "%d is %d.",
                            h, unassigned[h]);
            exit(1);
        }
    }

    // Adjust size of undefined spaces
    for (std::string space : elementary_spaces_priority_) {
        // Assign MOs to the undefined space with the highest priority
        if (not mo_spaces_.count(space)) {
            std::vector<MOInfo> vec_mo_info;
            mo_spaces_[space] = std::make_pair(unassigned, vec_mo_info);
            for (size_t h = 0; h < nirrep_; ++h) {
                unassigned[h] = 0;
            }
        }
    }
    if (unassigned.sum() != 0) {
        outfile->Printf("\n  There is an error in the definition of the "
                        "orbital spaces.  There are %d unassigned MOs.",
                        unassigned.sum());
        exit(1);
    }

    // Compute orbital mappings
    for (size_t h = 0, p_abs = 0; h < nirrep_; ++h) {
        size_t p_rel = 0;
        for (std::string space : elementary_spaces_) {
            size_t n = mo_spaces_[space].first[h];
            for (size_t q = 0; q < n; ++q) {
                size_t p_order = p_abs;
                // If a reordering array is provided, use it to determine the index
                if (reorder_.size() > 0) {
                    p_order = reorder_[p_order];
                }
                mo_spaces_[space].second.push_back(std::make_tuple(p_order, h, p_rel));
                p_abs += 1;
                p_rel += 1;
            }
        }
    }

    // Compute the MO to correlated MO mapping
    std::vector<size_t> vec(nmopi_.sum());
    std::iota(vec.begin(), vec.end(), 0);

    // Remove the frozen core/virtuals
    for (MOInfo& mo_info : mo_spaces_["FROZEN_DOCC"].second) {
        size_t removed_mo = std::get<0>(mo_info);
        vec.erase(std::remove(vec.begin(), vec.end(), removed_mo), vec.end());
    }
    for (MOInfo& mo_info : mo_spaces_["FROZEN_UOCC"].second) {
        size_t removed_mo = std::get<0>(mo_info);
        vec.erase(std::remove(vec.begin(), vec.end(), removed_mo), vec.end());
    }

    mo_to_cmo_.assign(nmopi_.sum(), 1000000000);
    for (size_t n = 0; n < vec.size(); ++n) {
        mo_to_cmo_[vec[n]] = n;
    }

    // Define composite spaces

    // Print the space information
    size_t label_size = 1;
    for (std::string space : elementary_spaces_) {
        label_size = std::max(space.size(), label_size);
    }

    int banner_width = label_size + 4 + 6 * (nirrep_ + 1);

    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
    outfile->Printf("\n    %s", std::string(label_size, ' ').c_str());
    for (size_t h = 0; h < nirrep_; ++h)
        outfile->Printf(" %5s", irrep_label(h).c_str());
    outfile->Printf("   Sum");
    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());

    for (std::string space : elementary_spaces_) {
        psi::Dimension& dim = mo_spaces_[space].first;
        outfile->Printf("\n    %-*s", label_size, space.c_str());
        for (size_t h = 0; h < nirrep_; ++h) {
            outfile->Printf("%6d", dim[h]);
        }
        outfile->Printf("%6d", dim.sum());
    }
    outfile->Printf("\n    %-*s", label_size, "Total");
    for (size_t h = 0; h < nirrep_; ++h) {
        outfile->Printf("%6d", nmopi_[h]);
    }
    outfile->Printf("%6d", nmopi_.sum());
    outfile->Printf("\n  %s", std::string(banner_width, '-').c_str());
}

std::pair<SpaceInfo, bool> MOSpaceInfo::read_mo_space(const std::string& space,
                                                      std::shared_ptr<ForteOptions> options) {
    bool read = false;
    psi::Dimension space_dim(nirrep_);
    std::vector<MOInfo> vec_mo_info;
    if (not options->exists(space)) {
        SpaceInfo space_info(space_dim, vec_mo_info);
        return std::make_pair(space_info, false);
    }
    size_t vec_size = options->get_int_vec(space).size();
    if (vec_size == nirrep_) {
        for (size_t h = 0; h < nirrep_; ++h) {
            space_dim[h] = options->get_int_vec(space)[h];
        }
        read = true;
        outfile->Printf("\n  Read options for space %s", space.c_str());
    } else if (vec_size > 0) {
        std::string msg = "\n  The size of space " + space + " (" + std::to_string(vec_size) +
                          ") does not match the number of irreducible representations (" +
                          std::to_string(nirrep_) + ").";
        outfile->Printf("\n%s", msg.c_str());
        throw std::runtime_error(msg);
    }
    SpaceInfo space_info(space_dim, vec_mo_info);
    return std::make_pair(space_info, read);
}

std::pair<SpaceInfo, bool>
MOSpaceInfo::read_mo_space_from_map(const std::string& space,
                                    std::map<std::string, std::vector<size_t>>& mo_space_map) {
    bool read = false;
    psi::Dimension space_dim(nirrep_);
    std::vector<MOInfo> vec_mo_info;

    // lookup the space
    auto it = mo_space_map.find(space);
    if (it != mo_space_map.end()) {
        const auto& dim = mo_space_map[space];
        if (dim.size() == nirrep_) {
            for (size_t h = 0; h < nirrep_; ++h) {
                space_dim[h] = dim[h];
            }
            read = true;
        } else {
            throw std::runtime_error("\n  The size of space vector does not match the number of "
                                     "irreducible representations.");
        }
    }
    SpaceInfo space_info(space_dim, vec_mo_info);
    return std::make_pair(space_info, read);
}

std::shared_ptr<MOSpaceInfo> make_mo_space_info(const psi::Dimension& nmopi,
                                                const std::string& point_group,
                                                std::shared_ptr<ForteOptions> options) {
    auto mo_space_info = std::make_shared<MOSpaceInfo>(nmopi, point_group);
    mo_space_info->read_options(options);
    mo_space_info->compute_space_info();
    return mo_space_info;
}

std::shared_ptr<MOSpaceInfo>
make_mo_space_info_from_map(const psi::Dimension& nmopi, const std::string& point_group,
                            std::map<std::string, std::vector<size_t>>& mo_space_map,
                            std::vector<size_t> reorder) {

    auto mo_space_info = std::make_shared<MOSpaceInfo>(nmopi, point_group);
    mo_space_info->set_reorder(reorder);
    mo_space_info->read_from_map(mo_space_map);
    mo_space_info->compute_space_info();
    return mo_space_info;
}

} // namespace forte
