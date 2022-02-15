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

#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

void SA_MRDSRG::diis_manager_init() {
    diis_manager_ = std::make_shared<DIISManager>(diis_max_vec_, "SA_MRDSRG DIIS",
                                                  DIISManager::RemovalPolicy::LargestError,
                                                  DIISManager::StoragePolicy::OnDisk);

    amp_ptrs_.clear();
    res_ptrs_.clear();

    std::vector<std::string> blocks{"ca",   "cv",   "av",   "ccaa", "ccav", "ccva",
                                    "ccvv", "caaa", "caav", "cava", "cavv", "acaa",
                                    "acav", "acva", "acvv", "aaav", "aava", "aavv"};

    std::vector<size_t> sizes(18);
    for (int i = 0; i < 18; ++i) {
        auto block = blocks[i];
        if (block.size() == 2) {
            sizes[i] = T1_.block(block).numel();
            amp_ptrs_.push_back(T1_.block(block).data().data());
            res_ptrs_.push_back(DT1_.block(block).data().data());
        } else {
            sizes[i] = T2_.block(block).numel();
            amp_ptrs_.push_back(T2_.block(block).data().data());
            res_ptrs_.push_back(DT2_.block(block).data().data());
        }
    }

    diis_manager_->set_error_vector_size(
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
        sizes[6], sizes[7], sizes[8], sizes[9], sizes[10], sizes[11],
        sizes[12], sizes[13], sizes[14], sizes[15], sizes[16], sizes[17]);

    diis_manager_->set_vector_size(
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
        sizes[6], sizes[7], sizes[8], sizes[9], sizes[10], sizes[11],
        sizes[12], sizes[13], sizes[14], sizes[15], sizes[16], sizes[17]);
}

void SA_MRDSRG::diis_manager_add_entry() {
    diis_manager_->add_entry(
        36, res_ptrs_[0], res_ptrs_[1], res_ptrs_[2], res_ptrs_[3], res_ptrs_[4], res_ptrs_[5],
        res_ptrs_[6], res_ptrs_[7], res_ptrs_[8], res_ptrs_[9], res_ptrs_[10], res_ptrs_[11],
        res_ptrs_[12], res_ptrs_[13], res_ptrs_[14], res_ptrs_[15], res_ptrs_[16], res_ptrs_[17],
        amp_ptrs_[0], amp_ptrs_[1], amp_ptrs_[2], amp_ptrs_[3], amp_ptrs_[4], amp_ptrs_[5],
        amp_ptrs_[6], amp_ptrs_[7], amp_ptrs_[8], amp_ptrs_[9], amp_ptrs_[10], amp_ptrs_[11],
        amp_ptrs_[12], amp_ptrs_[13], amp_ptrs_[14], amp_ptrs_[15], amp_ptrs_[16], amp_ptrs_[17]);
}

void SA_MRDSRG::diis_manager_extrapolate() {
    diis_manager_->extrapolate(
        18, amp_ptrs_[0], amp_ptrs_[1], amp_ptrs_[2], amp_ptrs_[3], amp_ptrs_[4], amp_ptrs_[5],
        amp_ptrs_[6], amp_ptrs_[7], amp_ptrs_[8], amp_ptrs_[9], amp_ptrs_[10], amp_ptrs_[11],
        amp_ptrs_[12], amp_ptrs_[13], amp_ptrs_[14], amp_ptrs_[15], amp_ptrs_[16], amp_ptrs_[17]);
}

void SA_MRDSRG::diis_manager_cleanup() {
    amp_ptrs_.clear();
    res_ptrs_.clear();
    diis_manager_->reset_subspace();
    diis_manager_->delete_diis_file();
}
} // namespace forte
