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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "dsrg_mem.h"

using namespace psi;

namespace forte {

DSRG_MEM::DSRG_MEM() { mem_avai_ = 0; }

DSRG_MEM::DSRG_MEM(int64_t mem_avai, std::map<char, size_t> label_to_size)
    : mem_avai_(mem_avai), label_to_size_(label_to_size) {}

void DSRG_MEM::add_entry(const std::string& des, const size_t& mem_use, bool subtract) {
    data_[des] = mem_use;
    if (subtract) {
        mem_avai_ -= mem_use;
    }
}

void DSRG_MEM::add_entry(const std::string& des, const std::string& labels, int multiple,
                         bool subtract) {
    size_t nele = 1;
    for (const char& label : labels) {
        if (label_to_size_.find(label) == label_to_size_.end()) {
            throw psi::PSIEXCEPTION("MO label not found.");
        }
        nele *= label_to_size_[label];
    }
    add_entry(des, nele * multiple * sizeof(double), subtract);
}

void DSRG_MEM::print(const std::string& name) {
    print_h2(name + " Memory Information");
    for (const auto& pair : data_) {
        std::string description = pair.first;
        auto xb_pair = to_xb(pair.second, 1);
        psi::outfile->Printf("\n    %-40s %8.2f %2s", description.c_str(), xb_pair.first,
                             xb_pair.second.c_str());
    }

    if (mem_avai_ < 0) {
        auto xb_pair = to_xb(-mem_avai_, 1);
        std::string error = "Not enough memory to compute " + name + " energy.";
        psi::outfile->Printf("\n  %s", error.c_str());
        psi::outfile->Printf("\n  Please increase memory by at least %.2f %s.",
                             1.024 * xb_pair.first, xb_pair.second.c_str());
        throw psi::PSIEXCEPTION(error);
    }
}
} // namespace forte
