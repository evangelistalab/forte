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

#include "psi4/libdiis/diismanager.h"
#include "ambit/blocked_tensor.h"

#include "mrdsrg.h"

using namespace psi;

namespace forte {

std::vector<double> MRDSRG::copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                          BlockedTensor& T2,
                                          const std::vector<std::string>& label2) {
    std::vector<double> out;

    for (const auto& block : label1) {
        out.insert(out.end(), T1.block(block).data().begin(), T1.block(block).data().end());
    }
    for (const auto& block : label2) {
        out.insert(out.end(), T2.block(block).data().begin(), T2.block(block).data().end());
    }

    return out;
}

size_t MRDSRG::vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                BlockedTensor& T2, const std::vector<std::string>& label2) {
    size_t total_elements = 0;
    for (const auto& block : label1) {
        total_elements += T1.block(block).numel();
    }
    for (const auto& block : label2) {
        total_elements += T2.block(block).numel();
    }
    return total_elements;
}

void MRDSRG::return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                             BlockedTensor& T2, const std::vector<std::string>& label2,
                             const std::vector<double>& data) {
    // test data
    std::map<std::string, size_t> num_elements;
    size_t total_elements = 0;

    for (const auto& block : label1) {
        size_t numel = T1.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }
    for (const auto& block : label2) {
        size_t numel = T2.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }

    if (data.size() != total_elements) {
        throw psi::PSIEXCEPTION("Number of elements in T1 and T2 do not match the bid data vector");
    }

    // transfer data
    for (const auto& block : label1) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T1.block(block).numel();
        std::vector<double> T1_this_block(start, end);
        T1.block(block).data() = T1_this_block;
    }
    for (const auto& block : label2) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T2.block(block).numel();
        std::vector<double> T2_this_block(start, end);
        T2.block(block).data() = T2_this_block;
    }
}

void MRDSRG::diis_manager_init() {
    diis_manager_ = std::make_shared<DIISManager>(diis_max_vec_, "MRDSRG DIIS",
                                                  DIISManager::RemovalPolicy::LargestError,
                                                  DIISManager::StoragePolicy::OnDisk);

    diis_manager_->set_error_vector_size(DT1_, DT2_);

    diis_manager_->set_vector_size(T1_, T2_);
}

void MRDSRG::diis_manager_add_entry() {
    diis_manager_->add_entry(DT1_, DT2_, T1_, T2_);
}

void MRDSRG::diis_manager_extrapolate() {
    diis_manager_->extrapolate(T1_, T2_);
}

void MRDSRG::diis_manager_cleanup() {
    diis_manager_->reset_subspace();
    diis_manager_->delete_diis_file();
}
} // namespace forte
